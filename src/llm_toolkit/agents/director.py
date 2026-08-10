import json
from typing import TYPE_CHECKING, final

from llm_toolkit.agents.character import CharacterSheet
from llm_toolkit.logging import get_logger

if TYPE_CHECKING:
    from llm_toolkit.agents.agent import Agent

logger = get_logger(__name__)


@final
class Director:
    """Coordinates a separate facilitator agent and a cast of character agents."""

    def __init__(
        self,
        agents: dict[str, "Agent"] = None,
        director_agent: "Agent" = None,
        ai_turns: int = 2,
    ):
        if ai_turns <= 0:
            raise ValueError("ai_turns must be greater than 0")

        self.agents = dict(agents or {})
        self.director_agent = director_agent
        self.ai_turns = ai_turns
        self.character_sheets: dict[str, CharacterSheet] = {}
        self.public_transcript: list[dict[str, str]] = []
        self.private_transcripts: dict[str, list[dict[str, str]]] = {}

        logger.info("Director initialized with %s agents", len(self.agents))

    def set_director_agent(self, agent: "Agent"):
        """
        Sets the agent used to coordinate character conversations.

        Args:
            `agent` (Agent): Agent that selects speakers and gives direction.
        """

        self.director_agent = agent
        logger.info("Director coordinator agent updated")

    def set_ai_turns(self, ai_turns: int):
        """
        Sets the number of AI turns generated after each public user turn.

        Args:
            `ai_turns` (int): Number of AI speaking turns.
        """

        if ai_turns <= 0:
            raise ValueError("ai_turns must be greater than 0")

        self.ai_turns = ai_turns
        logger.info("Director AI turns per user turn changed to %s", ai_turns)

    def add_agent(self, name: str, agent: "Agent"):
        """
        Adds an agent to the director.

        Args:
            `name` (str): Name used to reference the agent.
            `agent` (Agent): Agent to add.
        """

        self.agents[name] = agent
        logger.info("Agent %s added to director", name)

    def add_character(
        self,
        character: CharacterSheet,
        model="gpt-5",
        name: str = None,
        **agent_options,
    ):
        """
        Creates an agent from a character sheet and adds it to the director.

        Args:
            `character` (CharacterSheet): Character sheet used for the agent.
            `model` (str): Model used by the agent. Defaults to gpt-5.
            `name` (str): Optional director name. Defaults to the character name.
            `agent_options`: Additional options passed to Agent.

        Returns:
            Agent: The created agent.
        """

        from llm_toolkit.agents.agent import Agent

        agent = Agent(
            model=model,
            system_prompt=character.to_system_prompt(),
            **agent_options,
        )
        character_name = name or character.name
        self.add_agent(character_name, agent)
        self.character_sheets[character_name] = character
        return agent

    def remove_agent(self, name: str):
        """
        Removes an agent from the director.

        Args:
            `name` (str): Name of the agent to remove.
        """

        del self.agents[name]
        self.character_sheets.pop(name, None)
        self.private_transcripts.pop(name, None)
        logger.info("Agent %s removed from director", name)

    def public_turn(
        self,
        message: str,
        target: str = None,
        ai_turns: int = None,
    ) -> list[dict[str, str]]:
        """
        Runs a public user turn that every agent can hear.

        Args:
            `message` (str): User message shared with every agent.
            `target` (str): Optional agent the user addressed directly.
            `ai_turns` (int): Optional AI turn override for this user turn.

        Returns:
            list[dict[str, str]]: AI responses selected by the director agent.
        """

        if not self.agents:
            raise ValueError("At least one agent is required for a public turn")
        if target is not None and target not in self.agents:
            raise ValueError(f"Unknown target: {target}")

        turn_count = self.ai_turns if ai_turns is None else ai_turns
        if turn_count <= 0:
            raise ValueError("ai_turns must be greater than 0")

        recipient = target or "everyone"
        opening_message = f"User to {recipient}: {message}"
        context = self._get_facilitation_context(target, turn_count)
        responses = self.coordinate(
            list(self.agents),
            opening_message,
            turns=turn_count,
            context=context,
        )

        self.public_transcript.append(
            {
                "speaker": "User",
                "content": message,
                "target": recipient,
            }
        )
        self.public_transcript.extend(
            {
                "speaker": entry["speaker"],
                "content": entry["content"],
                "target": "public",
            }
            for entry in responses
        )
        logger.info(
            "Completed public user turn addressed to %s with %s AI turns",
            recipient,
            turn_count,
        )
        return responses

    def private_turn(self, name: str, message: str) -> str:
        """
        Runs a private one-to-one turn between the user and one agent.

        Args:
            `name` (str): Name of the agent receiving the private message.
            `message` (str): Private user message.

        Returns:
            str: The agent's private response.
        """

        if name not in self.agents:
            raise ValueError(f"Unknown target: {name}")

        self.whisper("User [private]", name, message)
        response = self.speak(name, listeners=[])
        transcript = self.private_transcripts.setdefault(name, [])
        transcript.extend(
            [
                {"speaker": "User", "content": message},
                {"speaker": name, "content": response},
            ]
        )
        logger.info("Completed private user turn with agent %s", name)
        return response

    def get_public_transcript(self) -> list[dict[str, str]]:
        """
        Gets the public conversation transcript.

        Returns:
            list[dict[str, str]]: Copies of public conversation entries.
        """

        return [dict(entry) for entry in self.public_transcript]

    def get_private_transcript(self, name: str) -> list[dict[str, str]]:
        """
        Gets the private conversation transcript for one agent.

        Args:
            `name` (str): Name of the private conversation agent.

        Returns:
            list[dict[str, str]]: Copies of private conversation entries.
        """

        return [dict(entry) for entry in self.private_transcripts.get(name, [])]

    def _get_facilitation_context(self, target: str, ai_turns: int) -> str:
        cast = []
        for name in self.agents:
            character = self.character_sheets.get(name)
            if character is None:
                cast.append(name)
            else:
                cast.append(character.to_system_prompt())

        history = [
            f"{entry['speaker']} to {entry['target']}: {entry['content']}"
            for entry in self.public_transcript
        ]
        target_instruction = (
            f"The user directly addressed {target}. For the first AI turn, you must "
            f"select {target}. Select later speakers according to the conversation."
            if target is not None
            else "The user addressed the full group."
        )
        return (
            f"Generate {ai_turns} total AI speaking turns for this user turn. "
            "Characters may respond to the user or to another character. Avoid choosing "
            "a speaker who has nothing new to add.\n\n"
            f"{target_instruction}\n\n"
            "Character sheets:\n"
            + "\n\n---\n\n".join(cast)
            + "\n\nPublic conversation so far:\n"
            + ("\n".join(history) if history else "No earlier public turns.")
        )

    def send_text(self, name: str, text: str):
        """
        Sends a text message to one agent.

        Args:
            `name` (str): Name of the agent to message.
            `text` (str): Text to send.
        """

        self.agents[name].send_text(text)
        logger.debug("Sent text message to agent %s", name)

    def whisper(self, from_agent: str, to_agent: str, text: str):
        """
        Sends a private text message between agents.

        Args:
            `from_agent` (str): Name of the sending agent.
            `to_agent` (str): Name of the receiving agent.
            `text` (str): Text to send.
        """

        message = f"{from_agent}: {text}"
        self.agents[to_agent].send_text(message)
        logger.debug("Agent %s whispered to agent %s", from_agent, to_agent)

    def overhear(self, from_agent: str, text: str):
        """
        Sends one agent's message to every other agent.

        Args:
            `from_agent` (str): Name of the speaking agent.
            `text` (str): Text to send.
        """

        for name, agent in self.agents.items():
            if name != from_agent:
                agent.send_text(f"{from_agent}: {text}")

        logger.debug("Agent %s was overheard by other agents", from_agent)

    def broadcast(self, text: str):
        """
        Sends a text message to every agent.

        Args:
            `text` (str): Text to send.
        """

        for agent in self.agents.values():
            agent.send_text(text)

        logger.debug("Broadcast text message to %s agents", len(self.agents))

    def speak(self, name: str, listeners: list[str] = None):
        """
        Gets one agent response and sends it to selected listeners.

        Args:
            `name` (str): Name of the speaking agent.
            `listeners` (list[str]): Optional listener names. Defaults to every other agent.

        Returns:
            str: The speaking agent's response text.
        """

        response = self.get_response(name)
        response_text = self._get_response_text(response)
        listener_names = listeners

        if listener_names is None:
            listener_names = [
                agent_name for agent_name in self.agents if agent_name != name
            ]

        for listener in listener_names:
            self.agents[listener].send_text(f"{name}: {response_text}")

        logger.info("Agent %s spoke to %s listeners", name, len(listener_names))
        return response_text

    def converse(
        self,
        speakers: list[str],
        opening_message: str,
        rounds: int = 1,
    ) -> list[dict[str, str]]:
        """
        Runs a bounded conversation between a sequence of agents.

        Args:
            `speakers` (list[str]): Agent names in speaking order.
            `opening_message` (str): Message used to begin the conversation.
            `rounds` (int): Number of turns each agent receives. Defaults to 1.

        Returns:
            list[dict[str, str]]: Conversation entries with speaker and content.
        """

        if len(speakers) < 2:
            raise ValueError("A conversation requires at least two speakers")
        if rounds <= 0:
            raise ValueError("rounds must be greater than 0")

        self.agents[speakers[0]].send_text(opening_message)
        transcript = []

        for _ in range(rounds):
            for index, speaker in enumerate(speakers):
                listener = speakers[(index + 1) % len(speakers)]
                content = self.speak(speaker, listeners=[listener])
                transcript.append({"speaker": speaker, "content": content})

        logger.info(
            "Completed %s conversation rounds between %s agents",
            rounds,
            len(speakers),
        )
        return transcript

    def coordinate(
        self,
        speakers: list[str],
        opening_message: str,
        turns: int = 4,
        context: str = None,
    ) -> list[dict[str, str]]:
        """
        Runs a conversation with speaker selection managed by a director agent.

        Args:
            `speakers` (list[str]): Agent names available to the director agent.
            `opening_message` (str): Message used to begin the conversation.
            `turns` (int): Maximum number of character turns. Defaults to 4.
            `context` (str): Optional context supplied to the director agent.

        Returns:
            list[dict[str, str]]: Generated conversation entries.
        """

        if self.director_agent is None:
            raise RuntimeError("A director agent is required to coordinate a conversation")
        if not speakers:
            raise ValueError("At least one speaker is required")
        if turns <= 0:
            raise ValueError("turns must be greater than 0")

        missing_speakers = [name for name in speakers if name not in self.agents]
        if missing_speakers:
            raise ValueError(f"Unknown speakers: {', '.join(missing_speakers)}")

        for speaker in speakers:
            self.agents[speaker].send_text(opening_message)

        transcript = []

        for turn_index in range(turns):
            prompt = self._get_director_prompt(
                speakers,
                opening_message,
                transcript,
                turn_index + 1,
                turns,
                context,
            )
            self.director_agent.send_text(prompt)
            decision = self._parse_director_decision(
                self._get_response_text(self.director_agent.get_response())
            )
            speaker = decision.get("speaker")

            if speaker not in speakers:
                raise ValueError(f"Director selected unknown speaker: {speaker}")

            instruction = decision.get("instruction")
            if instruction:
                self.agents[speaker].send_text(f"Director: {instruction}")

            content = self.speak(
                speaker,
                listeners=[name for name in speakers if name != speaker],
            )
            transcript.append({"speaker": speaker, "content": content})

        logger.info(
            "Completed a director-coordinated conversation with %s turns",
            turns,
        )
        return transcript

    def _get_director_prompt(
        self,
        speakers: list[str],
        opening_message: str,
        transcript: list[dict[str, str]],
        turn_number: int,
        total_turns: int,
        context: str = None,
    ) -> str:
        conversation = [f"User: {opening_message}"]
        conversation.extend(
            f"{entry['speaker']}: {entry['content']}" for entry in transcript
        )
        prompt = (
            "Select the next speaker and give them a brief private instruction. "
            "Return only a JSON object with string fields named speaker and instruction.\n\n"
            f"Available speakers: {', '.join(speakers)}\n\n"
            f"Turn: {turn_number} of {total_turns}\n\n"
            "Conversation:\n"
            + "\n".join(conversation)
        )

        if context:
            prompt += f"\n\nContext:\n{context}"

        return prompt

    def _parse_director_decision(self, response_text: str) -> dict[str, str]:
        start = response_text.find("{")
        end = response_text.rfind("}")
        if start < 0 or end < start:
            raise ValueError("Director response did not contain a JSON object")

        decision = json.loads(response_text[start : end + 1])
        if not isinstance(decision, dict):
            raise ValueError("Director response must be a JSON object")
        return decision

    def get_response(self, name: str):
        """
        Gets a response from one agent.

        Args:
            `name` (str): Name of the agent to query.

        Returns:
            The model response.
        """

        logger.info("Getting response from agent %s", name)
        return self.agents[name].get_response()

    def get_responses(self):
        """
        Gets responses from every agent.

        Returns:
            dict: Responses keyed by agent name.
        """

        responses = {}

        for name, agent in self.agents.items():
            responses[name] = agent.get_response()

        logger.info("Got responses from %s agents", len(responses))
        return responses

    def _get_response_text(self, response) -> str:
        if isinstance(response, str):
            return response

        choices = (
            response.get("choices") if isinstance(response, dict) else response.choices
        )
        choice = choices[0]
        message = choice.get("message") if isinstance(choice, dict) else choice.message
        content = message.get("content") if isinstance(message, dict) else message.content

        if content is not None:
            return str(content)

        audio = (
            message.get("audio")
            if isinstance(message, dict)
            else getattr(message, "audio", None)
        )
        if audio is not None:
            transcript = (
                audio.get("transcript")
                if isinstance(audio, dict)
                else audio.transcript
            )
            return str(transcript)

        return ""

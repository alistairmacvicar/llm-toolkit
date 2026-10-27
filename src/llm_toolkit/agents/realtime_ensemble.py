"""Persistent realtime peers with the same routing policy as the text Director."""

import asyncio
import base64
import json
import os
from copy import deepcopy
from time import perf_counter
from typing import final
from uuid import uuid4

from llm_toolkit.agents.realtime import RealtimeSession
from llm_toolkit.agents.hierarchy import validate_hierarchy
from llm_toolkit.agents.turn_policy import decision_schema, eligible_children, turn_instructions, validate_decision


@final
class RealtimeEnsemble:
    """A half-duplex ensemble: record, direct, stream a reply, then hand over.

    The supplied Director owns the durable text memories. Each realtime peer is
    seeded with only that agent's history. ``emit`` and ``wait_playback`` are async
    application callbacks; no browser, filesystem or audio device is required.
    Sessions reconnect from committed memories after a transport failure. An
    interrupted utterance is never automatically replayed to a provider.
    """

    def __init__(self, director, *, model, voices=None, emit=None, wait_playback=None,
                 session_factory=None, api_key=None, timeout=90, speaker_delay=0.3):
        self.director = director
        self.model = model
        self.voices = dict(voices or {})
        self.emit = emit or self._ignore
        self.wait_playback = wait_playback or self._ignore
        self.session_factory = session_factory
        self.api_key = api_key
        self.timeout = timeout
        self.speaker_delay = speaker_delay
        self.sessions = {}
        self.recording = None
        self.busy = False
        self._audio = bytearray()
        self._active_peer = None

    async def _ignore(self, *args):
        pass

    async def connect(self):
        """Connect each peer once, retaining sockets across successful turns."""
        coordinators = self.director.get_coordinators()
        if not coordinators or any(agent is None for agent in coordinators.values()):
            raise ValueError("A coordinator agent is required for realtime conversations")
        if not self.director.agents:
            raise ValueError("At least one character is required for realtime conversations")
        if set(coordinators).intersection(self.director.agents):
            raise ValueError("Coordinators and characters must have different names")
        if self.director.hierarchy:
            validate_hierarchy(self.director.hierarchy, self.director.root, self.director.agents)
        names = [*coordinators, *self.director.agents]
        try:
            results = await asyncio.gather(*(self._connect_peer(name) for name in names if name not in self.sessions), return_exceptions=True)
            for result in results:
                if isinstance(result, BaseException):
                    raise result
        except BaseException:
            await self.close()
            raise

    async def _connect_peer(self, name):
        if self.session_factory:
            session = self.session_factory(name)
        else:
            key = self.api_key or os.getenv("OPENAI_API_KEY")
            if not key:
                raise ValueError("Set OPENAI_API_KEY to use realtime voice")
            session = RealtimeSession("wss://api.openai.com/v1/realtime", self.model, api_key=key)
        self.sessions[name] = session
        await session.connect()
        is_character = name in self.director.agents
        agent = self.director.agents[name] if is_character else self.director.get_coordinators()[name]
        instructions = agent.system_prompt
        if not is_character:
            instructions += "\n" + self.director._get_facilitation_context(None, self.director.ai_turns)
            instructions += "\nDirector tree: " + json.dumps(self.director.hierarchy)
        await session.update_session(
            instructions=instructions, output_modalities=["audio"] if is_character else ["text"],
            audio={"input": {"format": {"type": "audio/pcm", "rate": 24000}, "turn_detection": None,
                             "transcription": {"model": "gpt-4o-mini-transcribe"}},
                   "output": {"format": {"type": "audio/pcm", "rate": 24000}, "voice": self.voices.get(name, "marin")}},
        )
        if is_character and agent.messages:
            await session.send_text("Your conversation memory (including your own earlier replies):\n" + json.dumps(agent.messages, ensure_ascii=False), create_response=False)
        # Wait for acknowledgement now so authentication/configuration errors are
        # reported before the user starts recording.
        async with asyncio.timeout(self.timeout):
            while True:
                event = await session.receive_event()
                self._check_error(event)
                if event["type"] == "session.updated":
                    break

    async def close(self):
        """Release every peer, including partially opened connections."""
        sessions, self.sessions = self.sessions, {}
        await asyncio.gather(*(s.close() for s in sessions.values()), return_exceptions=True)
        self.recording = None
        self._audio.clear()
        self._active_peer = None

    async def begin(self, *, target=None, private=False):
        """Start a new PCM24k utterance. A target forces the first public speaker."""
        if self.busy or self.recording is not None:
            raise RuntimeError("Wait for the current turn to finish")
        if target is not None and target not in self.director.agents:
            raise ValueError("Unknown character")
        if private and target is None:
            raise ValueError("Private voice needs a recipient")
        self.busy = True
        try:
            await self.connect()
            self._audio.clear()
            peer = target if private else self.director.root
            await self.sessions[peer].send_event({"type": "input_audio_buffer.clear"})
            self.recording = {"target": target, "private": private}
        except BaseException:
            await self.close()
            raise
        finally:
            self.busy = False

    async def append_audio(self, pcm):
        """Forward microphone chunks immediately; retain at most 90 seconds."""
        if self.recording is None:
            raise RuntimeError("Start recording first")
        if not isinstance(pcm, bytes) or len(pcm) % 2:
            raise ValueError("Audio must be mono PCM16 bytes")
        if len(self._audio) + len(pcm) > 24000 * 2 * 90:
            raise ValueError("Utterances are limited to 90 seconds")
        self._audio.extend(pcm)
        peer = self.recording["target"] if self.recording["private"] else self.director.root
        try:
            await self.sessions[peer].append_audio(pcm)
        except BaseException:
            await self.close()
            raise

    async def finish(self):
        """Commit an utterance and return a complete, audibly ordered turn record."""
        if self.recording is None or len(self._audio) < 4800:
            raise ValueError("Record at least 100 milliseconds of audio")
        selection, self.recording = self.recording, None
        self.busy = True
        started = perf_counter()
        record = {"id": uuid4().hex, "private_to": selection["target"] if selection["private"] else None,
                  "target": selection["target"], "replies": [], "decisions": [], "input_audio": bytes(self._audio)}
        try:
            if selection["private"]:
                peer = selection["target"]
                input_id = await self._commit_audio(peer)
                reply, transcript = await self._speak(peer, "Respond to the user's private message.", "neutral", record["id"], input_id=input_id)
                record["message"] = transcript
                record["replies"].append(reply)
            else:
                await self._commit_audio(self.director.root)
                for turn in range(self.director.ai_turns):
                    node = self.director.root
                    directions = []
                    while True:
                        candidates = eligible_children(self.director.hierarchy, node, self.director.agents,
                            excluded=[r["speaker"] for r in record["replies"]] if self.director.avoid_repeat else [],
                            target=selection["target"] if turn == 0 else None)
                        if not candidates:
                            return await self._commit(record, started)
                        allow_stop = self.director.optional_replies and turn > 0
                        decision = await self._direct(node, candidates, allow_stop, directions, record, turn)
                        record["decisions"].append({"director": node, "turn": turn + 1, **decision})
                        await self.emit({"type": "decision", **record["decisions"][-1]})
                        if turn == 0 and node == self.director.root:
                            record["message"] = decision["transcript"]
                        speaker = decision["speaker"]
                        if speaker is None:
                            return await self._commit(record, started)
                        directions.append(decision["instruction"])
                        if speaker in self.director.agents:
                            break
                        node = speaker
                    session = self.sessions[speaker]
                    if turn == 0:
                        await asyncio.gather(*(s.send_text("User to everyone: " + record["message"], create_response=False)
                                              for name, s in self.sessions.items() if name != speaker and name != self.director.root))
                        await session.append_audio(bytes(self._audio))
                        await self._commit_audio(speaker)
                    reply, _ = await self._speak(speaker, "\n".join(directions), decision["emotion"], record["id"],
                        public_context=self.director.public_reply_context(record["message"], record["replies"]))
                    record["replies"].append(reply)
                    await asyncio.gather(*(s.send_text(f"{speaker}: {reply['content']}", create_response=False)
                                           for name, s in self.sessions.items() if name != speaker))
                    if self.speaker_delay:
                        await asyncio.sleep(self.speaker_delay)
            return await self._commit(record, started)
        except BaseException:
            # Closing discards uncommitted audio/history; the next connection is
            # seeded from the last successful turn. Do not duplicate a response.
            try:
                await self.emit({"type": "interrupted", "turn_id": record["id"], "replies": record["replies"]})
            finally:
                await self.close()
            raise
        finally:
            self.busy = False
            self._audio.clear()

    async def _direct(self, name, candidates, allow_stop, directions, record, turn):
        schema = decision_schema(candidates, allow_stop=allow_stop)
        schema["properties"]["transcript"] = {"type": "string", "description": "Faithful transcript of the user's latest audio; preserve their language."}
        schema["required"].append("transcript")
        prompt = f"Return a routing decision using the direct_turn function. Eligible immediate children: {candidates}."
        prompt += f"\nParent directions: {directions}."
        if record.get("message"):
            prompt += "\nCurrent user message: " + record["message"]
        if record["replies"]:
            prompt += "\nReplies this turn: " + json.dumps([{k: v for k, v in r.items() if k != "audio"} for r in record["replies"]])
        prompt += "\n" + self.director.public_reply_context(record.get("message", "Use the current user audio."), record["replies"])
        prompt += "\n" + turn_instructions(self.director.ai_turns, turn, optional_replies=self.director.optional_replies)
        prompt += "\nAlso include the required faithful transcript field in the function arguments, including when stopping."
        session = self.sessions[name]
        instructions = self.director.get_coordinators()[name].system_prompt
        instructions += "\n" + self.director._get_facilitation_context(None, self.director.ai_turns)
        instructions += "\nDirector tree: " + json.dumps(self.director.hierarchy)
        instructions += "\n" + prompt
        # Routing snapshots apply to one response, rather than accumulating as
        # conversation items on persistent sockets.
        await session.request_response(output_modalities=["text"], instructions=instructions,
            tools=[{"type": "function", "name": "direct_turn", "description": "Select who speaks next and their delivery.", "parameters": schema}],
            tool_choice={"type": "function", "name": "direct_turn"})
        response, _, _ = await self._collect(name)
        calls = [item for item in response.get("output", []) if item.get("type") == "function_call" and item.get("name") == "direct_turn"]
        if len(calls) != 1:
            raise ValueError("Realtime director did not return one routing decision")
        raw = json.loads(calls[0]["arguments"])
        decision = validate_decision(raw, candidates, allow_stop=allow_stop)
        if not isinstance(raw.get("transcript"), str) or not raw["transcript"].strip():
            raise ValueError("Realtime director did not transcribe the utterance")
        decision["transcript"] = raw["transcript"]
        await session.send_event({"type": "conversation.item.create", "item": {
            "type": "function_call_output", "call_id": calls[0]["call_id"], "output": json.dumps(decision)}})
        return decision

    async def _commit_audio(self, name):
        await self.sessions[name].commit_audio(create_response=False)
        async with asyncio.timeout(self.timeout):
            while True:
                event = await self.sessions[name].receive_event()
                self._check_error(event)
                if event["type"] == "input_audio_buffer.committed":
                    return event["item_id"]

    async def _speak(self, name, instruction, emotion, turn_id, input_id=None, public_context=None):
        playback_id = uuid4().hex
        await self.emit({"type": "audio_start", "speaker": name, "emotion": emotion, "id": playback_id, "turn_id": turn_id})
        await self.sessions[name].request_response(output_modalities=["audio"],
            instructions=f"{self.director.agents[name].system_prompt}\nDirection: {instruction}\nSpeak with {emotion} delivery. Do not say speaker labels or stage directions."
                         + ("\n" + public_context if public_context else ""))
        response, pcm, transcript = await self._collect(name, playback_id=playback_id, input_id=input_id)
        text = "".join(part.get("transcript", part.get("text", "")) for item in response.get("output", []) for part in item.get("content", []))
        if not text.strip() or not pcm:
            raise ValueError("Realtime character returned no speech or transcript")
        await self.emit({"type": "audio_end", "id": playback_id, "speaker": name, "text": text})
        await asyncio.wait_for(self.wait_playback(playback_id), timeout=max(self.timeout, len(pcm) / 48000 + 15))
        return {"speaker": name, "content": text, "emotion": emotion, "audio": pcm}, transcript

    async def _collect(self, name, *, playback_id=None, input_id=None):
        response, transcript, pcm = None, "", bytearray()
        self._active_peer = name
        async with asyncio.timeout(self.timeout):
            while response is None or (input_id and not transcript):
                event = await self.sessions[name].receive_event()
                self._check_error(event)
                kind = event["type"]
                if kind == "conversation.item.input_audio_transcription.completed" and event.get("item_id") == input_id:
                    transcript = event.get("transcript", "")
                elif kind in {"response.output_audio.delta", "response.audio.delta"} and playback_id:
                    pcm.extend(base64.b64decode(event["delta"]))
                    await self.emit({"type": "audio", "id": playback_id, "data": event["delta"]})
                elif kind == "response.done":
                    response = event["response"]
                    if response.get("status") != "completed":
                        raise RuntimeError(f"Realtime response {response.get('status', 'failed')}: {response.get('status_details', {})}")
        self._active_peer = None
        return response, bytes(pcm), transcript

    @staticmethod
    def _check_error(event):
        if event.get("type") == "error":
            raise RuntimeError(event.get("error", {}).get("message", "Realtime provider error"))
        if event.get("type") == "conversation.item.input_audio_transcription.failed":
            raise RuntimeError("Realtime input transcription failed")

    async def _commit(self, record, started):
        record["elapsed_seconds"] = perf_counter() - started
        self.commit_record(self.director, record)
        await self.emit({"type": "turn", "record": record})
        return record

    @staticmethod
    def commit_record(director, record):
        """Mirror a completed realtime turn into ordinary Agent text histories."""
        private = record["private_to"]
        recipients = [private] if private else list(director.agents)
        for name in recipients:
            director.agents[name].send_text(f"User {'[private]' if private else 'to everyone'}: {record['message']}")
        entries = [{"speaker": "User", "content": record["message"], "target": private or record.get("target") or "everyone"}]
        for reply in record["replies"]:
            speaker = reply["speaker"]
            for name in recipients:
                agent = director.agents[name]
                if name == speaker:
                    agent.messages.append({"role": "assistant", "content": reply["content"]})
                else:
                    agent.send_text(f"{speaker}: {reply['content']}")
            director.emotions[speaker] = reply["emotion"]
            entries.append({k: v for k, v in reply.items() if k != "audio"} | {"target": private or "public"})
        if private:
            director.private_transcripts.setdefault(private, []).extend(entries)
        else:
            director.public_transcript.extend(entries)
            director.decisions.extend(deepcopy(record["decisions"]))

"""Exercise complete ensemble turns with deterministic local transport peers."""

import asyncio
import base64
import json
import unittest

from llm_toolkit.agents.director import Director
from llm_toolkit.agents.realtime_ensemble import RealtimeEnsemble


class MemoryAgent:
    def __init__(self):
        self.system_prompt = "Respond briefly."
        self.messages = []

    def send_text(self, text):
        self.messages.append({"role": "user", "content": text})


class Peer:
    def __init__(self, name):
        self.name = name
        self.events = asyncio.Queue()
        self.text = []
        self.audio = []
        self.closed = False
        self.transcript = "Hello everyone"
        self.fail_response = False
        self.commits = 0

    async def connect(self):
        return self

    async def close(self):
        self.closed = True

    async def update_session(self, **options):
        self.events.put_nowait({"type": "session.updated"})

    async def receive_event(self):
        return await self.events.get()

    async def send_event(self, event):
        pass

    async def send_text(self, text, **options):
        self.text.append(text)

    async def append_audio(self, pcm):
        self.audio.append(pcm)

    async def commit_audio(self, **options):
        self.commits += 1
        item_id = f"{self.name}-{self.commits}"
        self.events.put_nowait({"type": "input_audio_buffer.committed", "item_id": item_id})
        self.events.put_nowait({"type": "conversation.item.input_audio_transcription.completed",
                               "item_id": item_id, "transcript": self.transcript})

    async def request_response(self, **options):
        if self.fail_response:
            raise OSError("Transport disconnected")
        if options.get("tools"):
            candidates = options["tools"][0]["parameters"]["properties"]["speaker"]["enum"]
            decision = {"speaker": candidates[0], "instruction": "Reply briefly.",
                        "reason": "Eligible speaker", "emotion": "neutral", "transcript": self.transcript}
            output = [{"type": "function_call", "name": "direct_turn", "call_id": "routing",
                       "arguments": json.dumps(decision)}]
        else:
            self.events.put_nowait({"type": "response.output_audio.delta", "delta": base64.b64encode(b"\0\0" * 24).decode()})
            output = [{"type": "message", "content": [{"type": "audio", "transcript": f"Reply from {self.name}"}]}]
        self.events.put_nowait({"type": "response.done", "response": {"status": "completed", "output": output}})


class RealtimeEnsembleTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.director = Director({"Ada": MemoryAgent(), "Lin": MemoryAgent()},
                                 director_agent=MemoryAgent(), ai_turns=2)
        self.director.avoid_repeat = True
        self.peers = {}
        self.emitted = []
        self.played = []

        def factory(name):
            peer = Peer(name)
            self.peers[name] = peer
            return peer

        async def emit(event):
            self.emitted.append(event)

        async def played(playback_id):
            self.played.append(playback_id)

        self.ensemble = RealtimeEnsemble(self.director, model="test", session_factory=factory,
                                         emit=emit, wait_playback=played, speaker_delay=0, timeout=1)
        self.addAsyncCleanup(self.ensemble.close)

    async def test_public_turn_enforces_target_and_commits_ordered_replies(self):
        await self.ensemble.begin(target="Lin")
        await self.ensemble.append_audio(b"\0\0" * 2400)
        record = await self.ensemble.finish()
        self.assertEqual([reply["speaker"] for reply in record["replies"]], ["Lin", "Ada"])
        self.assertEqual(record["message"], "Hello everyone")
        self.assertEqual(len(self.director.public_transcript), 3)
        self.assertEqual(len(self.played), 2)
        self.assertEqual([e["type"] for e in self.emitted if e["type"].startswith("audio_")],
                         ["audio_start", "audio_end", "audio_start", "audio_end"])
        for agent in self.director.agents.values():
            self.assertIn("Reply from Ada", json.dumps(agent.messages))
            self.assertIn("Reply from Lin", json.dumps(agent.messages))
        self.assertFalse(self.ensemble.busy)
        self.assertIsNone(self.ensemble.recording)

    async def test_private_turn_is_not_sent_to_other_peers_or_public_history(self):
        await self.ensemble.begin(target="Ada", private=True)
        self.peers["Ada"].transcript = "A private concern"
        await self.ensemble.append_audio(b"\0\0" * 2400)
        record = await self.ensemble.finish()
        self.assertEqual(record["message"], "A private concern")
        self.assertEqual(self.director.public_transcript, [])
        self.assertEqual(self.director.agents["Lin"].messages, [])
        self.assertEqual(self.director.director_agent.messages, [])
        self.assertEqual(len(self.director.private_transcripts["Ada"]), 2)
        for name in ("Director", "Lin"):
            self.assertEqual(self.peers[name].text, [])
            self.assertEqual(self.peers[name].audio, [])

    async def test_failed_turn_closes_peers_without_committing_and_can_restart(self):
        await self.ensemble.begin(target="Ada")
        await self.ensemble.append_audio(b"\0\0" * 2400)
        self.peers["Ada"].fail_response = True
        failed_peers = list(self.peers.values())
        with self.assertRaises(OSError):
            await self.ensemble.finish()
        self.assertTrue(all(peer.closed for peer in failed_peers))
        self.assertEqual(self.director.public_transcript, [])
        self.assertTrue(all(not agent.messages for agent in self.director.agents.values()))
        self.assertEqual(self.ensemble.sessions, {})
        self.assertFalse(self.ensemble.busy)
        await self.ensemble.begin(target="Ada")
        await self.ensemble.append_audio(b"\0\0" * 2400)
        record = await self.ensemble.finish()
        self.assertEqual(len(record["replies"]), 2)

    async def test_failed_turn_delivery_restores_durable_history(self):
        async def fail_delivery(event):
            if event["type"] == "turn":
                raise OSError("Client disconnected")
        self.ensemble.emit = fail_delivery
        await self.ensemble.begin(target="Ada")
        await self.ensemble.append_audio(b"\0\0" * 2400)
        with self.assertRaisesRegex(OSError, "Client disconnected"):
            await self.ensemble.finish()
        self.assertEqual(self.director.public_transcript, [])
        self.assertEqual(self.director.decisions, [])
        self.assertEqual(self.director.emotions, {})
        self.assertTrue(all(not agent.messages for agent in self.director.agents.values()))
        self.assertEqual(self.ensemble.sessions, {})

    async def test_empty_peer_sessions_are_not_visible_during_connect(self):
        # Concurrent public connect calls should wait for the same initialization.
        started, release = asyncio.Event(), asyncio.Event()
        original = Peer.connect
        async def pause(peer):
            started.set()
            await release.wait()
            return await original(peer)
        from unittest.mock import patch
        with patch.object(Peer, "connect", pause):
            first = asyncio.create_task(self.ensemble.connect())
            await started.wait()
            second = asyncio.create_task(self.ensemble.connect())
            await asyncio.sleep(0)
            try:
                self.assertFalse(second.done())
            finally:
                release.set()
                await asyncio.gather(first, second)

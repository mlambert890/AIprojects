import asyncio
import base64
import mimetypes
import os
import re
import tempfile
from typing import Any, List, Optional

import aiofiles
import aiohttp

LMSTUDIO_URL = "http://192.168.1.203:1234/v1"


class Pipe:
    class Valves:
        CLASSIFIER_MODEL = "qwen2.5-1.5b-instruct"
        CHAT_MODEL = "qwen/qwen3-4b-thinking-2507"
        CODE_MODEL = "deepseek-coder-v2-lite-instruct"
        VISION_MODEL = "qwen/qwen3-vl-30b"
        REASONING_MODEL = "openai/gpt-oss-120b"
        LONG_THRESHOLD = 600

    def __init__(self):
        self.valves = self.Valves()
        self.name = "ModelRouter_VisionFixed"

    # ----------------------------
    # Emit helpers
    # ----------------------------
    def _safe_emit(self, emitter, payload):
        if emitter:
            try:
                asyncio.create_task(emitter(payload))
            except Exception:
                pass

    def _emit_debug(self, emitter, text):
        self._safe_emit(
            emitter,
            {
                "type": "status",
                "data": {"status": "debug", "description": text, "done": False},
            },
        )

    def _emit_chat_final(self, emitter, content=""):
        if emitter:
            self._safe_emit(
                emitter,
                {
                    "type": "chat:message",
                    "data": {"role": "assistant", "content": content},
                },
            )

    # ----------------------------
    # Utilities
    # ----------------------------
    def _extract_text(self, content: Any) -> str:
        if content is None:
            return ""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, str):
                    parts.append(item)
                elif isinstance(item, dict):
                    t = item.get("text") or item.get("content")
                    if isinstance(t, str):
                        parts.append(t)
            return "\n".join(parts)
        return str(content)

    def _contains_code(self, text: str) -> bool:
        if not text or not isinstance(text, str):
            return False

        # Normalize unicode quotes to ASCII
        t = text.replace("“", '"').replace("”", '"').replace("‘", "'").replace("’", "'")

        # 1. Markdown fenced code = absolute
        if "```" in t:
            return True

        # 2. Modern language indicators
        if re.search(r"\b(def|class|function|import|#include|var |let |const )\b", t):
            return True

        # 3. BASIC line numbers, e.g. "10 PRINT", "200 GOTO"
        if re.search(r"^\s*\d+\s+[A-Za-z]", t, re.MULTILINE):
            return True

        # 4. BASIC commands
        if re.search(
            r"\b(print|goto|gosub|input|next|for|end|poke|peek|dim|cls)\b",
            t,
            re.IGNORECASE,
        ):
            return True

        # 5. Semicolons or assignment or code-like punctuation
        if ";" in t or "=" in t:
            return True

        # 6. Presence of brackets or symbols typical in code
        if re.search(r"[{}()$begin:math:display$$end:math:display$<>]", t):
            return True

        # 7. Multiple consecutive symbols (= likely code)
        if re.search(r"[-+/*]{2,}", t):
            return True

        return False

    def _has_image(self, msg: dict) -> bool:
        content = msg.get("content")
        if isinstance(content, list):
            for c in content:
                if isinstance(c, dict) and c.get("type") in ("image_url", "image"):
                    return True
        for f in msg.get("files", []) or []:
            if f.get("mime", "").lower().startswith("image/"):
                return True
        for att in msg.get("attachments", []) or []:
            url = att.get("url", "")
            if isinstance(url, str) and re.search(
                r"\.(png|jpg|jpeg|gif|bmp|webp|svg)$", url, re.I
            ):
                return True
        return False

    def _extract_inline_images(self, msg: dict) -> List[dict]:
        images = []
        for c in msg.get("content", []):
            if isinstance(c, dict) and c.get("type") in ("image_url", "image"):
                url = c.get("image_url", {}).get("url") or c.get("url")
                if isinstance(url, str):
                    if url.startswith("data:image/"):
                        _, b64data = url.split(",", 1)
                        images.append({"type": "image", "image": b64data})
                    else:
                        images.append({"type": "image", "url": url})
        return images

    # ----------------------------
    # Attachments downloader
    # ----------------------------
    async def _download_attachments(self, messages: List[dict]) -> List[dict]:
        out = []
        candidates = []
        for m in messages:
            for a in m.get("attachments", []) or []:
                url, name = a.get("url"), a.get("name", "file")
                if url:
                    candidates.append((url, name))
        if not candidates:
            return out
        async with aiohttp.ClientSession() as session:
            tasks = [self._download_one(session, url, name) for url, name in candidates]
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, dict):
                    out.append(r)
        return out

    async def _download_one(
        self, session: aiohttp.ClientSession, url: str, name: str
    ) -> Optional[dict]:
        suffix = os.path.splitext(name)[1] or ""
        mime, _ = mimetypes.guess_type(name)
        mime = mime or "application/octet-stream"
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.close()
        try:
            async with session.get(
                url, timeout=aiohttp.ClientTimeout(total=10)
            ) as resp:
                if resp.status != 200:
                    os.unlink(tmp.name)
                    return None
                async with aiofiles.open(tmp.name, "wb") as f:
                    await f.write(await resp.read())
            return {
                "path": tmp.name,
                "name": name,
                "mime": mime,
                "type": "image" if mime.startswith("image/") else "file",
            }
        except Exception:
            try:
                os.unlink(tmp.name)
            except:
                pass
            return None

    # ----------------------------
    # LMStudio calls
    # ----------------------------
    async def _call_lmstudio_once(self, model: str, messages: List[dict]) -> str:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LMSTUDIO_URL}/chat/completions",
                json={"model": model, "messages": messages},
            ) as resp:
                try:
                    j = await resp.json()
                except Exception:
                    txt = await resp.text()
                    return f"__error__ {resp.status} {txt}"
                return j.get("choices", [{}])[0].get("message", {}).get("content", "")

    async def _lmstudio_stream(
        self, model: str, messages: List[dict], files: List[dict], emitter
    ) -> str:
        if model == self.valves.VISION_MODEL:
            all_images = []
            for msg in messages:
                all_images.extend(self._extract_inline_images(msg))
            for f in files:
                if f.get("mime", "").startswith("image/"):
                    async with aiofiles.open(f["path"], "rb") as fp:
                        content = await fp.read()
                        all_images.append(
                            {
                                "type": "image",
                                "image": base64.b64encode(content).decode(),
                            }
                        )

        payload = {"model": model, "messages": messages}
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{LMSTUDIO_URL}/chat/completions", json=payload
            ) as resp:
                if resp.status != 200:
                    txt = await resp.text()
                    error_msg = f"LMStudio error {resp.status}: {txt}"
                    self._emit_debug(emitter, error_msg)
                    return error_msg
                data = await resp.json()
                assistant_content = (
                    data.get("choices", [{}])[0].get("message", {}).get("content", "")
                )
                return assistant_content

    # ----------------------------
    # Main pipe
    # ----------------------------
    async def pipe(
        self,
        body: dict,
        __user__=None,
        __event_emitter__=None,
        __event_call__=None,
        __files__=None,
    ):
        messages = body.get("messages", []) or []
        if not messages:
            return {"error": "No messages provided"}

        user_msg = messages[-1]
        user_text = self._extract_text(user_msg.get("content"))
        self._emit_debug(
            __event_emitter__, f"Router started; preview: {user_text[:200]}"
        )

        if self._has_image(user_msg):
            chosen = self.valves.VISION_MODEL
            reason = "override:image"
            self._emit_debug(
                __event_emitter__, "Image detected: routing to vision model"
            )
        else:
            classifier_messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a strict router classifier. Respond with EXACTLY ONE WORD ONLY from: reasoning, vision, code, chat.\n"
                        "Use 'reasoning' only for multi-step logical thinking, math, planning.\n"
                        "Use 'chat' for short/simple conversation.\n"
                        "Use 'vision' only when images are present.\n"
                        "Use 'code' only when code/programming tasks are present.\n"
                        "No punctuation, no explanation, no other text."
                    ),
                },
                {"role": "user", "content": user_text},
            ]
            cls_resp = (
                (
                    await self._call_lmstudio_once(
                        self.valves.CLASSIFIER_MODEL, classifier_messages
                    )
                    or ""
                )
                .strip()
                .lower()
            )
            self._emit_debug(__event_emitter__, f"Classifier raw: {cls_resp[:200]}")

            if cls_resp in ("reasoning", "vision", "code", "chat"):

                # ✅ OVERRIDE: classifier mislabels code as reasoning/chat
                if cls_resp != "code" and self._contains_code(user_text):
                    chosen = self.valves.CODE_MODEL
                    reason = f"override:code_detected(classifier:{cls_resp})"
                    self._emit_debug(
                        __event_emitter__, f"Classifier override -> {reason}"
                    )

                else:
                    # Normal classifier mapping
                    chosen = {
                        "reasoning": self.valves.REASONING_MODEL,
                        "vision": self.valves.VISION_MODEL,
                        "code": self.valves.CODE_MODEL,
                        "chat": self.valves.CHAT_MODEL,
                    }[cls_resp]
                    reason = f"classifier:{cls_resp}"
            else:
                if self._contains_code(user_text):
                    chosen = self.valves.CODE_MODEL
                    reason = "heuristic:code"
                elif len(user_text) >= self.valves.LONG_THRESHOLD:
                    chosen = self.valves.REASONING_MODEL
                    reason = "heuristic:long"
                else:
                    chosen = self.valves.CHAT_MODEL
                    reason = "heuristic:chat"
                self._emit_debug(__event_emitter__, f"Classifier fallback -> {reason}")

        self._emit_debug(__event_emitter__, f"Chosen model: {chosen} ({reason})")

        files_list = __files__ or []
        try:
            downloaded = await self._download_attachments(messages)
            if downloaded:
                files_list.extend(downloaded)
                self._emit_debug(
                    __event_emitter__, f"Downloaded {len(downloaded)} attachments"
                )
        except Exception:
            self._emit_debug(
                __event_emitter__, "Attachment download failed (continuing)"
            )

        sanitized_msgs = []

        if chosen == self.valves.VISION_MODEL:
            # -------- VISION CASE --------
            content_blocks = []

            # 1. Add user text
            if user_text.strip():
                content_blocks.append({"type": "text", "text": user_text})

            # 2. Inline images
            inline_imgs = self._extract_inline_images(user_msg)
            for img in inline_imgs:
                if "image" in img:
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/png;base64,{img['image']}"
                            },
                        }
                    )
                elif "url" in img:
                    content_blocks.append(
                        {"type": "image_url", "image_url": {"url": img["url"]}}
                    )

            # 3. Downloaded attachments
            for f in files_list:
                if f.get("mime", "").startswith("image/"):
                    async with aiofiles.open(f["path"], "rb") as fp:
                        b = await fp.read()
                    b64 = base64.b64encode(b).decode()
                    content_blocks.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:{f['mime']};base64,{b64}"},
                        }
                    )

            sanitized_msgs = [{"role": "user", "content": content_blocks}]

        else:
            # -------- NON-VISION CASE --------
            # Just treat as plain text
            sanitized_msgs = [{"role": "user", "content": user_text}]

        response = await self._lmstudio_stream(
            chosen, sanitized_msgs, files_list, __event_emitter__
        )
        return response

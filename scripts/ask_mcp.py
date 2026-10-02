"""MCP daemon that lets Claude ask you something while you're away from the keyboard.

The question is spoken out loud with ToastTTS and shown in a macOS dialog; the
tool call blocks until you click or type an answer, then hands it to Claude.
It runs in the background (launchd keeps it alive and starts it at login), with
the voice loaded once and shared by every Claude session.

  .venv/bin/python scripts/ask_mcp.py --install       # start now and at every login
  claude mcp add --scope user --transport http ask-me http://127.0.0.1:8765/mcp

Logs: ~/Library/Logs/ask-me.log. Stop for good:
  launchctl bootout gui/$UID/com.toasttts.ask-me && rm ~/Library/LaunchAgents/com.toasttts.ask-me.plist
"""

import os
import plistlib
import subprocess
import sys
import threading
from pathlib import Path

from mcp.server.mcpserver import MCPServer

from toast.engine import ToastEngine

PORT = 8765
LABEL = "com.toasttts.ask-me"
PLIST = Path.home() / "Library/LaunchAgents" / f"{LABEL}.plist"
LOG = Path.home() / "Library/Logs/ask-me.log"

# argv: question, then options. 0 options: free text. 1-3: buttons plus an
# optional note. More: a list to pick from.
DIALOG = """on run argv
  set q to item 1 of argv
  set opts to rest of argv
  activate
  with timeout of 86400 seconds
    if (count of opts) > 3 then
      set r to choose from list opts with prompt q with title "Claude asks"
      if r is false then return "(dismissed)"
      return item 1 of r
    end if
    if (count of opts) is 0 then set opts to {"Send"}
    set r to display dialog q default answer "" buttons opts default button 1 with title "Claude asks"
    if (count of argv) is 1 then return text returned of r
    set reply to button returned of r
    if text returned of r is not "" then set reply to reply & " - note: " & text returned of r
    return reply
  end timeout
end run"""

mcp = MCPServer("ask-me")
_tts = None
_one_at_a_time = threading.Lock()  # one of you: sessions asking at once take turns


@mcp.tool()
def ask_user(question: str, options: list[str] | None = None) -> str:
    """Ask the user and wait for their answer, even when they're away from the
    keyboard: the question is spoken aloud and shown in a desktop dialog. Use
    it for approvals (options=["Approve", "Deny"]) or choices (options=["A",
    "B", "C"]); with no options the user types a free answer. With 1-3 options
    the user may also add a note. Blocks until answered."""
    with _one_at_a_time:
        spoken = question + (" Options: " + ", ".join(options) + "." if options else "")
        speech = _tts.speak(spoken)
        dialog = subprocess.run(["osascript", "-e", DIALOG, question, *(options or [])],
                                capture_output=True, text=True)
        speech.stop()
        _tts.close()  # don't leave the idle keep-awake tone running between questions
    if dialog.returncode:
        return f"(dialog failed: {dialog.stderr.strip()})"
    return dialog.stdout.strip() or "(empty answer)"


def install():
    PLIST.parent.mkdir(parents=True, exist_ok=True)
    PLIST.write_bytes(plistlib.dumps({
        "Label": LABEL,
        "ProgramArguments": [sys.executable, str(Path(__file__).resolve())],
        "RunAtLoad": True,
        "KeepAlive": True,
        "StandardOutPath": str(LOG),
        "StandardErrorPath": str(LOG),
    }))
    domain = f"gui/{os.getuid()}"
    subprocess.run(["launchctl", "bootout", f"{domain}/{LABEL}"], capture_output=True)  # reinstall = restart
    subprocess.run(["launchctl", "bootstrap", domain, str(PLIST)], check=True)
    print(f"running: http://127.0.0.1:{PORT}/mcp (log: {LOG})")


if __name__ == "__main__":
    if "--install" in sys.argv:
        install()
    else:
        _tts = ToastEngine()  # load the voice once, before the first question
        mcp.run("streamable-http", port=PORT)  # 127.0.0.1 only

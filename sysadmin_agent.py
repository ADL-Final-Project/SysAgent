import os
import re
import subprocess
import sys

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain.agents import create_agent

load_dotenv()


def _run(cmd, timeout=30):
    try:
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        output = (proc.stdout + proc.stderr).strip()
        return output if output else "(command produced no output)"
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s."
    except FileNotFoundError:
        return f"Error: '{cmd[0]}' is not installed or not on PATH."


def _sudo(cmd, timeout=30):
    if os.geteuid() != 0:
        cmd = ["sudo", "--non-interactive"] + cmd
    return _run(cmd, timeout=timeout)


def _validate_username(username):
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", username):
        return (
            f"Invalid username '{username}'. "
            "Must start with a lowercase letter or underscore, "
            "contain only [a-z0-9_-], and be at most 32 characters."
        )
    return None


def _validate_service_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9_@.\-]+", name):
        return f"Invalid service name '{name}'."
    return None

########################


@tool
def check_system_info():
    lines = [
        "=== Kernel / OS ===",
        _run(["uname", "-a"]),
        "\n=== Hostname ===",
        _run(["hostname", "--fqdn"]),
        "\n=== Uptime ===",
        _run(["uptime", "-p"]),
    ]
    return "\n".join(lines)


@tool
def check_disk_usage():
    return _run(["df", "-h", "--output=source,size,used,avail,pcent,target"])


@tool
def check_memory():
    return _run(["free", "-h"])


@tool
def list_users():
    raw = _run(["getent", "passwd"])
    users = []
    for line in raw.splitlines():
        fields = line.split(":")
        if len(fields) >= 7 and fields[2].isdigit() and int(fields[2]) >= 1000:
            users.append(
                f"  {fields[0]:<20} uid={fields[2]:<6} "
                f"home={fields[5]:<25} shell={fields[6]}"
            )
    return "Non-system users:\n" + "\n".join(users) if users else "No non-system users found."


@tool
def add_user(username):
    if err := _validate_username(username):
        return err
    return _sudo(["useradd", "--create-home", "--shell", "/bin/bash", username])


@tool
def delete_user(username):
    if err := _validate_username(username):
        return err
    return _sudo(["userdel", "--remove", username])


@tool
def set_user_password(username, password):
    if err := _validate_username(username):
        return err

    chpasswd_cmd = (
        ["sudo", "--non-interactive", "chpasswd"] if os.geteuid() != 0 else ["chpasswd"]
    )
    try:
        proc = subprocess.run(
            chpasswd_cmd,
            input=f"{username}:{password}",
            capture_output=True,
            text=True,
            timeout=15,
        )
        output = (proc.stdout + proc.stderr).strip()
        return output if output else "Password updated successfully."
    except subprocess.TimeoutExpired:
        return "Error: chpasswd timed out."
    except FileNotFoundError:
        return "Error: 'chpasswd' not found."


@tool
def update_packages():
    update_out = _sudo(["apt-get", "update", "-y"], timeout=120)
    upgrade_out = _sudo(["apt-get", "upgrade", "-y",
                        "--with-new-pkgs"], timeout=600)
    return f"--- apt-get update ---\n{update_out}\n\n--- apt-get upgrade ---\n{upgrade_out}"


@tool
def list_services(state="active"):
    allowed_states = {"active", "failed", "all"}
    if state not in allowed_states:
        return f"Invalid state '{state}'. Choose from: {', '.join(sorted(allowed_states))}."

    cmd = ["systemctl", "list-units",
           "--type=service", "--no-pager", "--no-legend"]
    if state != "all":
        cmd += [f"--state={state}"]
    return _run(cmd)


@tool
def manage_service(service_name, action):
    allowed_actions = {"disable", "enable",
                       "restart", "start", "status", "stop"}

    if action not in allowed_actions:
        return f"Invalid action '{action}'. Choose from: {', '.join(sorted(allowed_actions))}."
    if err := _validate_service_name(service_name):
        return err

    cmd = ["systemctl", action, service_name, "--no-pager"]
    return _run(cmd) if action == "status" else _sudo(cmd)


#####################################

TOOLS = [
    check_system_info,
    check_disk_usage,
    check_memory,
    list_users,
    add_user,
    delete_user,
    set_user_password,
    update_packages,
    list_services,
    manage_service,
]

SYSTEM_PROMPT = (
    "You are a Linux system administration assistant with access to tools "
    "that perform real operations on this host.\n\n"
    "Guidelines:\n"
    "- Always call the appropriate tool rather than guessing at system state.\n"
    "- Report tool output clearly and concisely; highlight any errors or warnings.\n"
    "- For multi-step tasks, execute each step in sequence and summarise the outcome.\n"
    "- If a request is ambiguous or potentially destructive, ask the user to confirm before acting.\n"
    "- Never fabricate command output."
)


def build_agent():
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        sys.exit(
            "Error: ANTHROPIC_API_KEY is not set.\n"
        )

    llm = ChatAnthropic(model="claude-sonnet-4-20250514", temperature=0)
    return create_agent(llm, TOOLS, system_prompt=SYSTEM_PROMPT)


def extract_last_ai_text(messages):
    for msg in reversed(messages):
        if isinstance(msg, AIMessage):
            if isinstance(msg.content, str):
                return msg.content
            texts = [
                block["text"]
                for block in msg.content
                if isinstance(block, dict) and block.get("type") == "text"
            ]
            return "\n".join(texts)
    return "(no response)"


####################################

if __name__ == "__main__":
    agent = build_agent()
    history = []

    print("SysAgent — type 'exit' or 'quit' to stop, 'clear' to reset history.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            print("Goodbye.")
            break
        if user_input.lower() == "clear":
            history.clear()
            print("Conversation history cleared.\n")
            continue

        history.append(HumanMessage(content=user_input))
        result = agent.invoke({"messages": history})

        history = result["messages"]

        reply = extract_last_ai_text(history)
        print(f"\nAgent: {reply}\n")

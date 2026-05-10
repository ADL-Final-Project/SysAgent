import os
import re
import subprocess
import sys

from dotenv import load_dotenv
from langchain_ollama import ChatOllama
from langchain.messages import AIMessage, HumanMessage
from langchain.tools import tool
from langchain.agents import create_agent

from langchain_community.document_loaders import DirectoryLoader, TextLoader, PyPDFLoader
# from langchain_community.document_loaders import UnstructuredMarkdownLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

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
        return f"Invalid username '{username}'."

    return None


def _validate_group_name(group):
    if not re.fullmatch(r"[a-z_][a-z0-9_-]{0,31}", group):
        return f"Invalid group name '{group}'."

    return None


def _validate_service_name(name):
    if not re.fullmatch(r"[a-zA-Z0-9_@.\-]+", name):
        return f"Invalid service name '{name}'."

    return None


def _validate_package_name(pkg):
    if not re.fullmatch(r"[a-z0-9][a-z0-9+\-.]+", pkg):
        return f"Invalid package name '{pkg}'."

    return None


#################################


@tool
def check_system_info():
    """Return OS version, kernel, hostname, and uptime."""
    return f'''
        === Kernel / OS ===
        {_run(["uname", "-a"])}

        === Hostname ===
        {_run(["hostname", "--fqdn"])}

        === Uptime ===
        {_run(["uptime", "-p"])}'''


@tool
def check_disk_usage():
    """Show disk usage for all mounted filesystems in human-readable form."""
    return _run(["df", "-h", "--output=source,size,used,avail,pcent,target"])


@tool
def check_memory():
    """Show current RAM and swap usage in human-readable form."""
    return _run(["free", "-h"])


@tool
def check_network():
    """Show network interfaces with their addresses and listening TCP/UDP sockets."""
    return f'''
        === Interfaces ===
        {_run(["ip", "-brief", "address"])}

        === Listening sockets ===
        {_run(["ss", "-tlnup"])}'''


@tool
def list_processes(filter=""):
    """
    List running processes. Optionally filter output to lines containing a keyword.

    Args:
        filter: Case-insensitive substring to grep for (e.g. 'nginx'). Leave blank for all.
    """
    raw = _run(["ps", "aux", "--sort=-%cpu"])

    if not filter:
        return raw

    header, *rows = raw.splitlines()
    matched = [row for row in rows if filter.lower() in row.lower()]

    if not matched:
        return f"No processes matching '{filter}'."

    return header + "\n" + "\n".join(matched)


@tool
def show_logs(service, lines=50):
    """
    Show the most recent journald log entries for a systemd service.

    Args:
        service: Name of the systemd service (e.g. 'nginx', 'ssh').
        lines: Number of log lines to return (default 50).
    """
    if err := _validate_service_name(service):
        return err

    return _run(["journalctl", "-u", service, "-n", str(lines), "--no-pager"])


@tool
def install_packages(packages):
    """
    Install one or more apt packages. Runs 'apt-get update' first to ensure
    the index is fresh.

    Args:
        packages: Space-separated list of package names to install (e.g. 'curl git htop').
    """
    pkg_list = packages.split()

    for pkg in pkg_list:
        if err := _validate_package_name(pkg):
            return err

    update_out = _sudo(["apt-get", "update", "-y"], timeout=120)

    install_out = _sudo(
        ["apt-get", "install", "-y"] + pkg_list,
        timeout=300,
    )

    return f'''
    --- apt-get update ---
    {update_out}
    
    --- apt-get install ---
    {install_out}'''


@tool
def remove_packages(packages, purge=False):
    """
    Remove one or more apt packages.

    Args:
        packages: Space-separated list of package names to remove.
        purge: If True, also delete configuration files (apt-get purge).
    """
    pkg_list = packages.split()

    for pkg in pkg_list:
        if err := _validate_package_name(pkg):
            return err

    action = "purge" if purge else "remove"

    return _sudo(["apt-get", action, "-y"] + pkg_list, timeout=120)


@tool
def update_packages():
    """
    Refresh the apt package index and upgrade all installed packages.
    This may take several minutes depending on the number of pending updates.
    """
    update_out = _sudo(["apt-get", "update", "-y"], timeout=120)
    upgrade_out = _sudo(["apt-get", "upgrade", "-y",
                        "--with-new-pkgs"], timeout=600)

    return f'''
    --- apt-get update ---
    {update_out}
    
    --- apt-get upgrade ---
    {upgrade_out}'''


@tool
def list_users():
    """List all non-system users (UID >= 1000) with login name, UID, home, and shell."""
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
    """
    Create a new Linux user account with a home directory and bash as the default shell.

    Args:
        username: Login name for the new user.
    """
    if err := _validate_username(username):
        return err

    return _sudo(["useradd", "--create-home", "--shell", "/bin/bash", username])


@tool
def delete_user(username):
    """
    Delete a Linux user account and remove their home directory.

    Args:
        username: Login name of the user to delete.
    """
    if err := _validate_username(username):
        return err

    return _sudo(["userdel", "--remove", username])


@tool
def set_user_password(username, password):
    """
    Set the password for an existing Linux user.

    Args:
        username: Login name of the target user.
        password: The new plaintext password (sent to chpasswd via stdin, never via argv).
    """
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
def list_groups():
    """List all groups on the system with their members."""
    raw = _run(["getent", "group"])
    lines = []

    for line in raw.splitlines():
        fields = line.split(":")

        if len(fields) >= 4:
            members = fields[3] if fields[3] else "(none)"

            lines.append(
                f"  {fields[0]:<20} gid={fields[2]:<6} members={members}")

    return "Groups:\n" + "\n".join(lines) if lines else "No groups found."


@tool
def add_user_to_groups(username, groups):
    """
    Add an existing user to one or more supplementary groups.

    Args:
        username: Login name of the target user.
        groups: Comma-separated list of group names (e.g. 'sudo,docker,www-data').
    """
    if err := _validate_username(username):
        return err

    group_list = [g.strip() for g in groups.split(",") if g.strip()]

    if not group_list:
        return "Error: no groups provided."

    for group in group_list:
        if err := _validate_group_name(group):
            return err

    return _sudo(["usermod", "-aG", ",".join(group_list), username])


@tool
def remove_user_from_group(username, group):
    """
    Remove a user from a single supplementary group.

    Args:
        username: Login name of the target user.
        group: Name of the group to remove the user from.
    """
    if err := _validate_username(username):
        return err
    if err := _validate_group_name(group):
        return err

    return _sudo(["gpasswd", "--delete", username, group])


@tool
def list_services(state="active"):
    """
    List systemd services filtered by their current state.

    Args:
        state: One of 'active', 'failed', or 'all'.
    """
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
    """
    Perform an action on a systemd service.

    Args:
        service_name: Name of the service, e.g. 'nginx' or 'ssh'.
        action: One of 'start', 'stop', 'restart', 'status', 'enable', 'disable'.
    """
    allowed_actions = {"disable", "enable",
                       "restart", "start", "status", "stop"}

    if action not in allowed_actions:
        return f"Invalid action '{action}'. Choose from: {', '.join(sorted(allowed_actions))}."

    if err := _validate_service_name(service_name):
        return err

    cmd = ["systemctl", action, service_name, "--no-pager"]

    # status -> read-only
    return _run(cmd) if action == "status" else _sudo(cmd)


@tool
def firewall_status():
    """Show the current ufw firewall status and all active rules."""
    return _sudo(["ufw", "status", "verbose"])


@tool
def firewall_enable(enabled):
    """
    Enable or disable the ufw firewall.

    Args:
        enabled: True to enable ufw, False to disable it.
    """
    action = "enable" if enabled else "disable"

    return _sudo(["ufw", "--force", action])


@tool
def firewall_allow(port_or_service, proto="any", direction="in", comment=""):
    """
    Add a ufw ALLOW rule.

    Args:
        port_or_service: Port number, range (e.g. '8000:8080'), or service name (e.g. 'ssh').
        proto: Protocol — 'tcp', 'udp', or 'any' (default).
        direction: 'in' (default) or 'out'.
        comment: Optional free-text comment attached to the rule.
    """
    cmd = ["ufw"]

    if direction == "out":
        cmd += ["allow", "out"]
    else:
        cmd += ["allow", "in"]

    spec = port_or_service if proto == "any" else f"{port_or_service}/{proto}"
    cmd.append(spec)

    if comment:
        cmd += ["comment", comment]

    return _sudo(cmd)


@tool
def firewall_deny(port_or_service, proto="any", direction="in"):
    """
    Add a ufw DENY rule.

    Args:
        port_or_service: Port number, range, or service name.
        proto: Protocol — 'tcp', 'udp', or 'any' (default).
        direction: 'in' (default) or 'out'.
    """
    cmd = ["ufw"]

    if direction == "out":
        cmd += ["deny", "out"]
    else:
        cmd += ["deny", "in"]

    spec = port_or_service if proto == "any" else f"{port_or_service}/{proto}"
    cmd.append(spec)

    return _sudo(cmd)


@tool
def firewall_delete_rule(rule_number):
    """
    Delete a ufw rule by its number (as shown in 'firewall_status').

    Args:
        rule_number: Integer rule number to delete.
    """
    return _sudo(["ufw", "--force", "delete", str(rule_number)])


@tool
def firewall_reset():
    """Reset ufw to its default state, removing all rules. Requires confirmation from the user."""
    return _sudo(["ufw", "--force", "reset"])


###########################


TOOLS = [
    check_system_info,
    check_disk_usage,
    check_memory,
    check_network,
    list_processes,
    show_logs,
    install_packages,
    remove_packages,
    update_packages,
    list_users,
    add_user,
    delete_user,
    set_user_password,
    list_groups,
    add_user_to_groups,
    remove_user_from_group,
    list_services,
    manage_service,
    firewall_status,
    firewall_enable,
    firewall_allow,
    firewall_deny,
    firewall_delete_rule,
    firewall_reset,
]

SYSTEM_PROMPT = '''
    You are SysAgent, a Linux system administration assistant with access to tools 
    that perform real operations on this host.

    Guidelines:
    - Always call the appropriate tool rather than guessing at system state.
    - Before performing administration tasks, call search_knowledge_base to check for relevant runbooks, procedures, or documentation.
    - Report tool output clearly and concisely; highlight any errors or warnings.
    - For multi-step tasks, execute each step in sequence and summarise the outcome.
    - If a request is ambiguous or potentially destructive, ask the user to confirm before acting.
    - Never fabricate command output.'''


def build_agent():
    ollama_url = os.getenv("OLLAMA_BASE_URL")
    ollama_model = os.getenv("OLLAMA_LANG_MODEL")
    embed_model = os.getenv("OLLAMA_EMBED_MODEL")
    docs_path = os.getenv("DOCS_PATH")

    if not ollama_url:
        sys.exit("Error: OLLAMA_BASE_URL is not set.\n")
    if not ollama_model:
        sys.exit("Error: OLLAMA_LANG_MODEL is not set.\n")
    if not embed_model:
        sys.exit("Error: OLLAMA_EMBED_MODEL is not set.\n")
    if not docs_path:
        sys.exit("Error: DOCS_PATH is not set.\n")

    retriever = build_retriever(ollama_url, embed_model, docs_path)

    @tool
    def search_knowledge_base(query):
        """
        Search the local documentation knowledge base for relevant information.
        Use this when the user asks about procedures, runbooks, or anything
        that may be documented locally.

        Args:
            query: A natural language description of what to look up.
        """
        results = retriever.invoke(query)

        if not results:
            return "No relevant documents found in the knowledge base."

        retr = "\n\n---\n\n".join(
            f"[{doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
            for doc in results
        )

        return retr

    llm = ChatOllama(model=ollama_model, base_url=ollama_url, temperature=0)

    return create_agent(llm, [search_knowledge_base] + TOOLS, system_prompt=SYSTEM_PROMPT)


def build_retriever(ollama_url, embed_model, docs_path):
    print("Processing documents for retrieval...\n")

    if not os.path.isdir(docs_path):
        sys.exit(
            f"Error: DOCS_PATH '{docs_path}' does not exist or is not a directory.")

    loaders = [
        DirectoryLoader(docs_path, glob="**/*.txt",
                        loader_cls=TextLoader, silent_errors=True),
        # DirectoryLoader(docs_path, glob="**/*.md",
        #                 loader_cls=UnstructuredMarkdownLoader, silent_errors=True),
        DirectoryLoader(docs_path, glob="**/*.pdf",
                        loader_cls=PyPDFLoader, silent_errors=True),
    ]

    docs = []

    for loader in loaders:
        docs.extend(loader.load())

    if not docs:
        print(
            f"Warning: no documents found in '{docs_path}'.")

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=150
    ).split_documents(docs)

    embeddings = OllamaEmbeddings(model=embed_model, base_url=ollama_url)

    vectorstore = Chroma.from_documents(chunks, embeddings)

    return vectorstore.as_retriever(search_kwargs={"k": 4})


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

import sys
from pathlib import Path
from src.pipeline import Pipeline

WORKSPACE = "workspace"
BANNER = """
╔══════════════════════════════════════════╗
║              AMLAgent                    ║
║    Autonomous ML Pipeline Agent          ║
╚══════════════════════════════════════════╝
"""


def list_files():
    ws = Path(WORKSPACE)
    ws.mkdir(exist_ok=True)
    return [f.name for f in ws.iterdir() if f.is_file()]


def parse_input(user_input: str):
    files = list_files()
    for f in files:
        if f.lower() in user_input.lower():
            return f, user_input
    return None


def main():
    print(BANNER)
    pipeline = Pipeline(workspace=WORKSPACE, max_iterations=10)

    while True:
        files = list_files()
        if files:
            print("Workspace files:")
            for f in files:
                print(f"  \u00b7 {f}")
        else:
            print("Workspace is empty. Drop your dataset into workspace/")
        print()

        try:
            user_input = input("You > ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye.")
            sys.exit(0)

        if not user_input or user_input.lower() in ("exit", "quit", "q"):
            print("Bye.")
            sys.exit(0)

        result = parse_input(user_input)
        if not result:
            print("Say something like: I have heart.csv, predict heart disease\n")
            continue

        file_name, task = result
        print(f"\nStarting pipeline on {file_name}...\n")

        try:
            output = pipeline.run(file_name, task)
            if output["success"]:
                print(f"Done. Best val score: {output['best_val_score']}")
            else:
                print(f"Failed: {output['error']}")
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()

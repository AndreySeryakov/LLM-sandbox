#!/usr/bin/env python3
import sys
from economic_analysis import EconomicAnalyzer
from rich.console import Console
from rich.table import Table
from rich.prompt import Prompt


def interactive_menu(analyzer):
    console = Console()
    last_choice = None
    while True:
        if last_choice not in ("4", "5"):
            console.clear()
        console.rule("[bold cyan]Econ Analyzer[/bold cyan]")
        table = Table(show_header=False, box=None)
        options = [
            ("1", "List rounds"),
            ("2", "View round details"),
            ("3", "View agent messages"),
            ("4", "Agent participation summary"),
            ("5", "Round participation summary"),
            ("6", "Round memory notes"),
            ("7", "Round strategy feedback"),
            ("8", "Round general feedback"),
            ("9", "Search messages"),
#            ("10", "Debug agent init"),
#            ("11", "Plot round messages"),
            ("0", "Exit"),
        ]
        for key, desc in options:
            table.add_row(f"[bold]{key}[/bold]", desc)
        console.print(table)
        choice = Prompt.ask("Select an option", choices=[o[0] for o in options], default="0")
        if choice == "0":
            console.print("Goodbye!", style="bold green")
            break

        # Execute selected action with error handling
        try:
            if choice == "1":
                rounds = analyzer.get_rounds()
                console.print("Available rounds:", style="bold yellow")
                for r in rounds:
                    console.print(f"- {r}")

            elif choice == "2":
                rid = Prompt.ask("Enter round ID to view")
                # Fetch structured details
                details = analyzer.get_round_details(rid)
                console.print(f"[bold cyan]=== DETAILS FOR ROUND {rid} ===[/bold cyan]")
                console.print(f"Round: {details.get('round')}")
                console.print(f"Timestamp: {details.get('timestamp')}")
                console.print(f"Database: {details.get('database')}")
                agents = details.get('agents', [])
                if agents:
                    tbl = Table(title="Participating Agents")
                    tbl.add_column("Name")
                    tbl.add_column("Memory Note")
                    tbl.add_column("Specializations")
                    for ag in agents:
                        tbl.add_row(
                            ag.get('name', ''),
                            ag.get('memory_note', ''),
                            ", ".join(ag.get('specializations', []))
                        )
                    console.print(tbl)
                else:
                    console.print("No agents participated in this round.", style="bold yellow")

            elif choice == "3":
                aid = Prompt.ask("Enter agent ID")
                rid = Prompt.ask("Enter round ID")
                analyzer.get_agent_round_messages(int(aid), rid)

            elif choice == "4":
             #   aid = Prompt.ask("Enter agent ID")
                analyzer.agent_participation_summary()

            elif choice == "5":
                #rid = Prompt.ask("Enter round ID")
                analyzer.round_participation_summary()

            elif choice == "6":
                rid = Prompt.ask("Enter round ID")
                analyzer.get_round_memory_notes(rid)

            elif choice == "7":
                rid = Prompt.ask("Enter round ID")
                analyzer.get_round_strategy_feedback(rid)

            elif choice == "8":
                rid = Prompt.ask("Enter round ID")
                analyzer.get_round_general_feedback(rid)

            elif choice == "9":
                term = Prompt.ask("Enter search term")
                analyzer.search_agent_messages(term)

            elif choice == "10":
                aid = Prompt.ask("Enter agent ID")
                analyzer.debug_agent_init(int(aid))

            elif choice == "11":
                rid = Prompt.ask("Enter round ID")
                out = Prompt.ask("Enter output path for plot (leave blank to skip saving)", default="")
                out_path = out if out else None
                analyzer.plot_round_messages(rid, out_path)
                if out_path:
                    console.print(f"Plot saved to [bold]{out_path}[/bold]")
                else:
                    console.print("Plot completed (not saved).")

            else:
                console.print("Invalid choice, please select again.", style="bold red")
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {str(e)}")
        finally:
            Prompt.ask("Press Enter to continue")
            last_choice = choice



def main():
    console = Console()
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "conversation_logs/conversation_logs.db"
    try:
        analyzer = EconomicAnalyzer(db_path)
    except Exception as e:
        console.print(f"[bold red]Failed to initialize analyzer:[/bold red] {e}")
        sys.exit(1)

    interactive_menu(analyzer)

if __name__ == "__main__":
    main()

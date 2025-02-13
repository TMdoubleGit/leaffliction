import os


def get_terminal_size():
    """Permet d'obtenir la taille du terminal"""
    try:
        columns, rows = os.get_terminal_size(0)
        return columns, rows
    except OSError:
        print(OSError)
        return None


def ft_tqdm(lst: range):
    """Recoit une liste en argument et affiche une barre
     de progression en fonction du nombre de valeur dans la liste"""
    total = len(lst)
    terminal_size = get_terminal_size()

    for i, item in enumerate(lst):
        progress = min(1.0, (i + 1) / total)
        percentage = int(progress * 100)
        counter = f"{i + 1}/{total}"
        gap_size = terminal_size[0] - len(str(percentage)) - len(counter) - 30
        bar_length = int(progress * gap_size)
        bar = "█" * bar_length

        percentage_str = f"{percentage}%"
        bar_str = f"|{bar:<{gap_size}}|"
        progress_str = f"{i + 1}/{total}"

        output_str = f"{percentage_str}{bar_str} {progress_str}"

        print(f"\r{output_str}", end="", flush=True)
        yield item
    print()

def print_banner(text: str) -> None:
    try:
        from pyfiglet import Figlet

        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        fig = Figlet(font="slant")
        ascii_art = fig.renderText("WORLDSCORE")
        print(ascii_art)
        fig = Figlet(font="small")
        ascii_art = fig.renderText(text)
        print(ascii_art)
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
    except ImportError:
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )
        print("WORLDSCORE")
        print(text)
        print(
            "=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-=-="
        )

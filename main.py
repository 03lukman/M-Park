import tkinter as tk
from src.handler import MParkingApp
from src.ui import UI

def main():
    root = tk.Tk()
    app = MParkingApp(root)
    app.ui = UI(app)

    app.update_streams()
    app.update_table()

    root.mainloop()

if __name__ == "__main__":
    main()

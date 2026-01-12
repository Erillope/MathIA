class Logger():
  def __init__(self, name: str) -> None:
    self.name = name
    self.log("")

  def log(self, message: str) -> None:
    print(f"[{self.name}]: {message}")
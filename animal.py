class Animal:
    """Base class representing a generic animal."""
    def __init__(self, name: str):
        self.name = name

    def speak(self) -> str:
        raise NotImplementedError("Subclasses must implement 'speak' method")

    def __str__(self) -> str:
        return f"{self.__class__.__name__} named {self.name}"


class Dog(Animal):
    """Dog subclass implementing speak."""
    def speak(self) -> str:
        return "Woof!"

    def fetch(self, item: str) -> str:
        return f"{self.name} fetched the {item}."


class Cat(Animal):
    """Cat subclass implementing speak."""
    def speak(self) -> str:
        return "Meow!"

    def scratch(self) -> str:
        return f"{self.name} is scratching the furniture."

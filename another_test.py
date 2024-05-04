from contextlib import nullcontext


class MyClass:
    def __init__(self, name: str) -> None:
        self.name = name

    def __call__(self, updated_name: str):
        self.name = updated_name
        if self.name == "null":
            return nullcontext()
        return self

    def __enter__(self):
        print("Entering context manager")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        print("Exiting context manager")
        return self


if __name__ == "__main__":
    mc = MyClass("test")
    with mc("null"):
        print(mc.name)
        mc("updated_name")
        print(mc.name)

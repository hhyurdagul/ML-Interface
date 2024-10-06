
from PySide6.QtWidgets import QLineEdit


def is_float(num):
    try:
        float(num)
        return True
    except ValueError:
        return False

class QIntLineEdit(QLineEdit):
    def __init__(self, text: str, min_value: int, max_value: int):
        super().__init__(text)
        self.textEdited.connect(
            lambda x: self.setText(str(min_value)) if not x.isnumeric()
            else self.setText(str(max_value)) if int(x) > max_value else None
        )

class QFloatLineEdit(QLineEdit):
    def __init__(self, text: str, min_value: float, max_value: float):
        super().__init__(text)
        self.textEdited.connect(
            lambda x: self.setText(str(min_value)) if not is_float(x)
            else self.setText(str(max_value)) if float(x) > max_value else None
        )

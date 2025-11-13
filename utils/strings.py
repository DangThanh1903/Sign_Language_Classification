
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên",
        "C" : "C",
        "O" : "O",
        "Toi": "Tôi",
        "Ten": "Tên",
        "T" : "T",
        "H" : "H",
        "A" : "A",
        "N" : "N",
        "dau_huyen" : "dau_huyen"
    }

    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]

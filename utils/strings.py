
class ExpressionHandler:

    MAPPING = {
        "bình_thường": "Ngồi yên",
        "cảm_ơn": "Cảm ơn",
        "xin_chào": "Xin chào",
        "xin_chào2": "Xin chào",
        "không": "Không",
        "Không2": "Không",
        "Sorry": "Xin lỗi",
        "at": "Ở",
        "Tên": "Tên",
        "gap_ban": "Gặp bạn",
        "Tôi": "Tôi",
        "1": "Một",
        "2": "Hai",
        "3": "Ba",
        "4": "Bốn",
        "5": "Năm",
        "age": "Tuổi",
        "vui": "Vui",
        "Like": "OK",
        "'": "Sắc",
        "^": "Mũ",
        "A": "A",
        "B": "B",
        "C": "C",
        "C_01": "C",
        "D": "D",
        "Đ": "Đ",
        "E": "E",
        "F": "F",
        "G": "G",
        "H": "H",
        "I": "I",
        "J": "J",
        "K": "K",
        "L": "L",
        "M": "M",
        "N": "N",
        "O": "O",
        "O_01": "O",
        "P": "P",
        "Q": "Q",
        "R": "R",
        "S": "S",
        "T": "T",
        "U": "U",
        "V": "V",
        "W": "W",
        "X": "X",
        "Y": "Y",
        "Dau Á": "Dấu Á",
        "Dau Huyen": "Dấu Huỳn",
        "Dau Sac_01": "Dấu Sắc",
        "Dau Sac_02": "Dấu Sắc",
        "Chao": "Chào",
        "Toi": "Tôi"
    }

    def __init__(self):
        # Save the current message and the time received the current message
        self.current_message = ""

    def receive(self, message):
        self.current_message = message

    def get_message(self):
        return ExpressionHandler.MAPPING[self.current_message]

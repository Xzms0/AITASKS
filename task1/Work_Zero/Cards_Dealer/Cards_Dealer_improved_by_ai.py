import random
from pathlib import Path


CARDS = {
    "Hearts": ['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
    "Spades": ['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
    "Diamonds": ['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
    "Clubs": ['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
    "Joker": ['Big_Joker','Little_Joker']
}

# 牌面权重（可根据游戏规则调整）
CARDS_VALUE = {
    "Big_Joker": 17, "Little_Joker": 16,
    "K": 13, "Q": 12, "J": 11, "10": 10,
    "9": 9, "8": 8, "7": 7, "6": 6, "5": 5,
    "4": 4, "3": 3, "2": 15, "A": 14
}


def build_deck():
    """构造一副牌（列表形式），每张牌表示为 (suit, rank) 或单字符串 Joker。"""
    deck = []
    for suit in ("Hearts", "Spades", "Diamonds", "Clubs"):
        for rank in CARDS[suit]:
            deck.append((suit, rank))
    # Joker 单独添加
    deck.append(("Joker", "Big_Joker"))
    deck.append(("Joker", "Little_Joker"))
    return deck


def card_value(card):
    """返回卡牌权重，用于排序。card 是 (suit, rank) 的元组。"""
    _, rank = card
    return CARDS_VALUE.get(rank, 0)


def format_card(card):
    """把 card 转为字符串行，例如：Hearts:A"""
    suit, rank = card
    return f"{suit}:{rank}\n"


def deal(deck=None):
    """发牌：返回字典 {'player1': [...], 'player2': [...], 'player3': [...], 'others': [...]}。
    函数内部不使用外部可变状态。
    """
    if deck is None:
        deck = build_deck()
    # 随机洗牌
    random.shuffle(deck)

    results = {"player1": [], "player2": [], "player3": [], "others": []}

    # 发前 51 张给三位玩家，每人 17 张
    for i in range(51):
        card = deck[i]
        if i % 3 == 0:
            results["player1"].append(card)
        elif i % 3 == 1:
            results["player2"].append(card)
        else:
            results["player3"].append(card)

    # 剩余 3 张为 others
    for i in range(51, 54):
        results["others"].append(deck[i])

    # 根据权重排序（降序）
    for k in results:
        results[k].sort(key=card_value, reverse=True)

    return results


def write_results(results_dict, out_dir=None):
    """将结果写入 out_dir/results/*.txt。确保目录存在，使用清晰的字符串格式写入。
    返回写入的文件路径列表。
    """
    if out_dir is None:
        out_dir = Path(__file__).absolute().parent / "results"
    out_dir.mkdir(parents=True, exist_ok=True)

    written = []
    for player, cards in results_dict.items():
        txt = out_dir / f"{player}.txt"
        with open(txt, "w", encoding="utf-8") as f:
            for card in cards:
                f.write(format_card(card))
        written.append(txt)
    return written


if __name__ == "__main__":
    deck = build_deck()
    results = deal(deck)
    files = write_results(results)
    print("发牌完成，已写入：")
    for p in files:
        print(f" - {p}")

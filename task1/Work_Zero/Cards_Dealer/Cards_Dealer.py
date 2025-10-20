import random
from pathlib import Path

results={"player1":[],"player2":[],"player3":[],"others":[]}

CARDS={"Hearts":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Spades":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Diamonds":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Clubs":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Joker":['Big_Joker','Little_Joker']}

CARDS_VALUE={"Big_Joker":17,"Little_Joker":16,
             "K":13,"Q":12,"J":11,"10":10,
             "9":9,"8":8,"7":7,"6":6,"5":5,
             "4":4,"3":3,"2":15,"A":14}

def choose(num):
    if num<=13:suit="Hearts"
    elif num<=26:suit="Spades"
    elif num<=39:suit="Diamonds"
    elif num<=52:suit="Clubs"
    else:suit="Joker"
    
    name=CARDS[suit][(num-1)%13]
    return [suit,":",name,"\n"]
    

def dealer():
    cards_list=random.sample(range(1,55), 54)

    for i in range(51):
        if i%3==0:results["player1"].append(choose(cards_list[i]))
        elif i%3==1:results["player2"].append(choose(cards_list[i]))
        elif i%3==2:results["player3"].append(choose(cards_list[i]))

    for i in range(51,54):
        results["others"].append(choose(cards_list[i]))

    sort_key=lambda tmp : CARDS_VALUE[tmp[2]]
    for player in list(results.keys()):
        results[player].sort(key=sort_key,reverse=True)

    root_dir=Path(__file__).absolute().parent/"results"
    for player in list(results.keys()):
        txt=root_dir/f"{player}.txt"
        with open(txt,"w") as file:
            for line in results[txt.stem]:
                file.writelines(line)

if __name__ == "__main__":
    dealer()
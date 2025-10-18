import random

player1=[]
player2=[]
player3=[]
others=[]

CARDS={"Hearts":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Spades":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Diamands":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Clubs":['A','2','3','4','5','6','7','8','9','10','J','Q','K'],
       "Joker":['Big_Joker','Little_Joker']}

CARDS_VALUE={"Big_Joker":17,"Little_Joker":16,
             "K":13,"Q":12,"J":11,"10":10,
             "9":9,"8":8,"7":7,"6":6,"5":5,
             "4":4,"3":3,"2":15,"A":14}

def choose(num):
    if num<=13:suit="Hearts"
    elif num<=26:suit="Spades"
    elif num<=39:suit="Diamands"
    elif num<=52:suit="Clubs"
    else:suit="Joker"
    name=CARDS[suit][(num-1)%13]
    return [suit,":",name,"\n"]
    

def dealer():
    cards_list=random.sample(range(1,55), 54)

    for i in range(51):
        if i%3==0:player1.append(choose(cards_list[i]))
        elif i%3==1:player2.append(choose(cards_list[i]))
        elif i%3==2:player3.append(choose(cards_list[i]))
    for i in range(51,54):
        others.append(choose(cards_list[i]))

    sort_key=lambda tmp : CARDS_VALUE[tmp[2]]
    player1.sort(key=sort_key,reverse=True)
    player2.sort(key=sort_key,reverse=True)
    player3.sort(key=sort_key,reverse=True)
    others.sort(key=sort_key,reverse=True)

dealer()

'''print(player1)
print(player2)
print(player3)
print(other)'''

with open("task1/Work_Zero/Cards_Dealer/result/player1.txt","w") as file1:
    for i in player1:
        file1.writelines(i)

with open("task1/Work_Zero/Cards_Dealer/result/player2.txt","w") as file1:
    for i in player2:
        file1.writelines(i)

with open("task1/Work_Zero/Cards_Dealer/result/player3.txt","w") as file1:
    for i in player3:
        file1.writelines(i)

with open("task1/Work_Zero/Cards_Dealer/result/others.txt","w") as file1:
    for i in others:
        file1.writelines(i)



from __future__ import annotations

class MyZoo:

    def __init__(self,animals={}):
        self.animals=animals
        print("My Zoo!")
        self.animals_list=list(self.animals.keys())
        self.animals_number=list(self.animals.values())

    def __str__(self):
        zoo_list=""
        for i in self.animals_list:
            j=self.animals[i]
            zoo_list+=f"{i}-{j} "
        return zoo_list

    def __len__(self):
        return sum(self.animals_number)
    
    def __eq__(self, another:MyZoo):
        self.animals_list.sort()
        another.animals_list.sort()
        if self.animals_list == another.animals_list:
            return True
        return False
    
    def __ne__(self, another:MyZoo):
        self.animals_list.sort()
        another.animals_list.sort()
        if self.animals_list != another.animals_list:
            return True
        return False

if __name__ == "__mian__":
    my_zoo1=MyZoo({"dog":5,"cat":7})
    my_zoo2=MyZoo({"cat":3,"dog":10})
    print(f"Zoo1's Len: {len(my_zoo1)}")
    print(f"Zoo2's Len: {len(my_zoo2)}")
    print(f"Zoo1's Print: {my_zoo1}")
    print(f"Zoo2's Print: {my_zoo2}")
    print(f"Zoo1 == Zoo2: {my_zoo1==my_zoo2}")
    print(f"Zoo1 != Zoo2: {my_zoo1!=my_zoo2}")
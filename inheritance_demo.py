class Pet:
    def __init__(self,name,age):
        self.name = name 
        self.age = age
        self.owner = "Me"

    def __str__(self) -> str:
        return f"my pet's name is {self.name} and he is {self.age} years old"

#add new method
class MyPet(Pet):
    def print_owner(self):
        return f"the owner of my {self.name} is {self.owner}"

a = Pet("Doggie",1)
print(a)

b = MyPet("doggie",2)
print(b.print_owner())
        
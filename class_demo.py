class Pet:
    def __init__(self,name,age):
        self.name =name
        self.age = age
        self.owner = "Me"
        self.animal = "dog"
    def get_name(self):
        return self.name
    
    def get_owner(self):
        return self.owner
    
    def __str__(self) -> str:
        return f"{self.name} is a {self.animal} and is {self.age} years old"
    
    def print_string(self):
        return f"{self.name} hello"
p = Pet("Pedro",1)   

print(p)
print(p.get_name())

print(f"hello {p.get_name()} and your owner is {p.get_owner()}")

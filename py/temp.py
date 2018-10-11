import mymodule as mm

# was loading module successful? 
a = mm.Foo()

a.add_msg("This is a message! Please remain calm.")
a.add_msg("Second msg coming through!")
a.add_msg("Okay, last message for now")

# print the entire message
print(a)

# or use a method
a.get_msg()

# documentation
help(mm.Foo)

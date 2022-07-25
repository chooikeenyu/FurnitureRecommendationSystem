import pandas as pd
def jaccard_similarity(x,y):
    intersection_cardianality = len(set.intersection(*[set(x),set(y)]))
    union_cardiality = len(set.union(*[set(x),set(y)]))
    return intersection_cardianality/float(union_cardiality)

# -------------------------------BATHROOM FURNITURE-------------------------------
bathroom_furniture = pd.read_excel(r'dataset/Bathroom Furniture.xlsx')
new_bathroom_furniture = bathroom_furniture.dropna()
new_bathroom_furniture = new_bathroom_furniture.reset_index(drop=True)

title_name = new_bathroom_furniture.loc[:,"Title"] # get only the title

# for i,v in title_name.items(): # access the index and items
#     similarity_score = jaccard_similarity()

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

# print("LEN OF LIST OF DUPLICATE")
# print(len(list_of_duplicate))           # All duplicates
# for item in list_of_duplicate:
#     print(item,end=" ")

mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_bathroom_furniture = new_bathroom_furniture.drop(mylist)

new_bathroom_furniture = new_bathroom_furniture.sample(n=400)
print(new_bathroom_furniture)
new_bathroom_furniture.to_excel(r'dataset/cleaned_Bathroom Furniture.xlsx',index=False)

# print("LEN OF LIST OF NEW LIST")
# print(len(mylist))
# for item in mylist:
#     print(item,end=" ")

# example1 = title_name.iloc[mylist]
# example1 = example1.to_frame()
# example1.sort_values("Title")


# -------------------------------BEDROOM FURNITURE-------------------------------

bedroom_furniture = pd.read_excel(r'dataset/Bedroom Furniture.xlsx')

new_bedroom_furniture = bedroom_furniture.dropna()
new_bedroom_furniture = new_bedroom_furniture.reset_index(drop=True)
title_name = new_bedroom_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_bedroom_furniture = new_bedroom_furniture.drop(mylist)

new_bedroom_furniture = new_bedroom_furniture.sample(n=400)
print(new_bedroom_furniture)
new_bedroom_furniture.to_excel(r'dataset/cleaned_Bedroom Furniture.xlsx',index=False)


# -------------------------------DINING ROOM FURNITURE-------------------------------

diningroom_furniture = pd.read_excel(r'dataset/DiningRoom Furniture.xlsx')

new_diningroom_furniture = diningroom_furniture.dropna()
new_diningroom_furniture = new_diningroom_furniture.reset_index(drop=True)

title_name = new_diningroom_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_diningroom_furniture = new_diningroom_furniture.drop(mylist)

new_diningroom_furniture = new_diningroom_furniture.sample(n=400)
print(new_diningroom_furniture)
new_diningroom_furniture.to_excel(r'dataset/cleaned_Diningroom Furniture.xlsx',index=False)

# -------------------------------GAME AND RECREATION ROOM FURNITURE-------------------------------

gameroom_furniture = pd.read_excel(r'dataset/GameAndRecreationRoom Furniture.xlsx')

new_gameroom_furniture = gameroom_furniture.dropna()
new_gameroom_furniture = new_gameroom_furniture.reset_index(drop=True)
title_name = new_gameroom_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_gameroom_furniture = new_gameroom_furniture.drop(mylist)

new_gameroom_furniture = new_gameroom_furniture.sample(n=400)
print(new_gameroom_furniture)
new_gameroom_furniture.to_excel(r'dataset/cleaned_Gameroom Furniture.xlsx',index=False)

# -------------------------------HOME OFFICE FURNITURE-------------------------------

homeoffice_furniture = pd.read_excel(r'dataset/HomeOffice Furniture.xlsx')

new_homeoffice_furniture = homeoffice_furniture.dropna()
new_homeoffice_furniture = new_homeoffice_furniture.reset_index(drop=True)

title_name = new_homeoffice_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_homeoffice_furniture_furniture = new_homeoffice_furniture.drop(mylist)

new_homeoffice_furniture = new_homeoffice_furniture.sample(n=400)
print(new_homeoffice_furniture)
new_homeoffice_furniture.to_excel(r'dataset/cleaned_HomeOffice Furniture.xlsx',index=False)

# -------------------------------KIDS FURNITURE-------------------------------

kids_furniture = pd.read_excel(r'dataset/Kids Furniture.xlsx')

new_kids_furniture = kids_furniture.dropna()
new_kids_furniture = new_kids_furniture.reset_index(drop=True)

title_name = new_kids_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_kids_furniture = new_kids_furniture.drop(mylist)

new_kids_furniture = new_kids_furniture.sample(n=400)
print(new_kids_furniture)
new_kids_furniture.to_excel(r'dataset/cleaned_Kids Furniture.xlsx',index=False)

# -------------------------------KITCHEN FURNITURE-------------------------------

kitchen_furniture = pd.read_excel(r'dataset/Kitchen Furniture.xlsx')

new_kitchen_furniture = kitchen_furniture.dropna()
new_kitchen_furniture = new_kitchen_furniture.reset_index(drop=True)

title_name = new_kitchen_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_kitchen_furniture = new_kitchen_furniture.drop(mylist)

new_kitchen_furniture = new_kitchen_furniture.sample(n=400)
print(new_kitchen_furniture)
new_kitchen_furniture.to_excel(r'dataset/cleaned_Kitchen Furniture.xlsx',index=False)

# -------------------------------LIVING ROOM FURNITURE-------------------------------

livingroom_furniture = pd.read_excel(r'dataset/LivingRoom Furniture.xlsx')

new_livingroom_furniture = livingroom_furniture.dropna()
new_livingroom_furniture = new_livingroom_furniture.reset_index(drop=True)

title_name = new_livingroom_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()


mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_livingroom_furniture = new_livingroom_furniture.drop(mylist)

new_livingroom_furniture = new_livingroom_furniture.sample(n=400)
print(new_livingroom_furniture)
new_livingroom_furniture.to_excel(r'dataset/cleaned_Livingroom Furniture.xlsx',index=False)

# -------------------------------REPLACEMENT PARTS-------------------------------

replacementparts_furniture = pd.read_excel(r'dataset/Replacement Parts.xlsx')

new_replacementparts_furniture = replacementparts_furniture.dropna()
new_replacementparts_furniture = new_replacementparts_furniture.reset_index(drop=True)

title_name = new_replacementparts_furniture.loc[:,"Title"] # get only the title

temp_of_duplicate = []
list_of_duplicate = []

for i in range(len(title_name)): # 670
    for y in range(len(title_name)): # 0-670
        # if y != range(len(title_name))[-1]: # if y not equal to 670
        if y != i:
            similarity_score = jaccard_similarity(title_name.iloc[i],title_name.iloc[y])
            if similarity_score > 0.90:
                temp_of_duplicate.append(y)
            if y == len(title_name)-1:
                temp_of_duplicate.sort()
                if temp_of_duplicate:
                    temp_of_duplicate.pop(0)
    list_of_duplicate.extend(temp_of_duplicate)
    temp_of_duplicate.clear()

mylist.clear()
mylist = list(dict.fromkeys(list_of_duplicate))     # after removing duplicate

new_replacementparts_furniture = new_replacementparts_furniture.drop(mylist)

new_replacementparts_furniture = new_replacementparts_furniture.sample(n=400)
print(new_replacementparts_furniture)
new_replacementparts_furniture.to_excel(r'dataset/cleaned_Replacementparts Furniture.xlsx',index=False)


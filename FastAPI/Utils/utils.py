def split_list(lst, rate):
    index = int(len(lst) * rate)
    list1 = lst[:index]
    list2 = lst[index:]
    
    if not list1:
        list1 = list2[-1:]
        list2 = list2[:-1]
    elif not list2:
        list2 = list1[-1:]
        list1 = list1[:-1]
    
    return list1, list2

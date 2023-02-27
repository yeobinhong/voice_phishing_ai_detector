from hanspell import spell_checker
def spell_check(input):
    input_convert = input.replace('.', '.#').split('#') #문장 단위로 분리
    input_list = [""]
    for i in input_convert:
        if (len(input_list[-1]) + len(i)) < 500:
            input_list[-1] += i
        else:
            input_list.append(i)
    result = spell_checker.check(input_list)
    result = ''
    for j, k in enumerate(input_list):
        a = spell_checker.check([input_list[j]])
        a = a[0].checked
        result = result + a
    return result

#https://vincinotes.com/%ED%8C%8C%EC%9D%B4%EC%8D%AC-py-hanspell%EB%A1%9C-%EB%84%A4%EC%9D%B4%EB%B2%84-%EB%A7%9E%EC%B6%A4%EB%B2%95-%EA%B2%80%EC%82%AC%EA%B8%B0-%EC%82%AC%EC%9A%A9%ED%95%98%EA%B8%B0/#i
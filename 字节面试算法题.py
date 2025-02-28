def func(n, A):
    print(n, A)

    n_str = str(n)
    length = len(n_str)
    A = sorted(A, reverse=True)
    d = {i:A for i in range(length)}
    ans = []
    flag = False

    i = 0
    while i < length:
        current_num = int(n_str[i])
        
        if len(d[i]) == 0:
            return -1
        
        print(current_num, ans, d[i], i)
        if current_num < A[-1]:
            if len(ans) > 0:
                d[i-1].remove(int(ans[-1]))
                ans = ans[:-1]
                i -= 1
            else:
                return int(''.join([str(A[0])]*(length-len(ans)-1)))
            print('callback')
            continue

        for a in d[i]:
            print('--', a)
            if current_num > a:
                ans.append(str(a))
                flag = True
                break
            elif current_num == a:
                ans.append(str(a))
                break

        if flag:
            ans.extend([str(A[0])]*(length-len(ans)))
            break

        i += 1

    return int(''.join(ans)) if len(ans) else -1


print(func(22, [2]))
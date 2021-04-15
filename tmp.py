class Solution:
    def myAtoi(self, s: str) -> int:
        num_dict = {}
        for i in range(10):
            num_dict[str(i)] = i
        s = s.lstrip()
        signal = 1

        keys = list(num_dict.keys()) + ['+', '-']
        if len(s) == 0 or (len(s) > 0 and s[0] not in keys):
            return 0

        if s[0] == '-':
            boundry = (1 << 31)
            signal = -1
            s = s[1:]
        elif s[0] == '+':
            boundry = (1 << 31) - 1
            s = s[1:]
        else:
            boundry = (1 << 31) - 1

        if len(s) == 0:
            return 0

        num = 0
        for i in s:
            if i not in num_dict.keys():
                break
            num = num * 10 + num_dict[i]
            if num > boundry:
                return signal * boundry
        return signal * num

    def threeSum(self, nums):
        if nums is None or len(nums) < 3:
            return []

        result = []
        nums = sorted(nums)

        if nums[0] > 0 or nums[-1] < 0:
            return []

        n = len(nums)
        for i in range(n-1):
            if nums[i] > 0:
                break

            if i > 0 and nums[i] == nums[i-1]:
                continue

            L = i + 1
            R = n - 1

            while L < R:
                three_sum = nums[i] + nums[L] + nums[R]

                if three_sum == 0:
                    result.append([nums[i], nums[L], nums[R]])
                    while L < n-1 and nums[L] == nums[L+1]:
                        L += 1
                    while R > 0 and nums[R] == nums[R-1]:
                        R -= 1
                    L += 1
                    R -= 1
                elif three_sum > 0:
                    R -= 1
                else:
                    L += 1
        return result

    def letterCombinations(self, digits):
        start = 'a'
        digital_letter = {}
        for i in range(2, 10, 1):
            if i == 7 or i == 9:
                digital_letter[str(i)] = [start, chr(ord(start) + 1), chr(ord(start) + 2), chr(ord(start) + 3)]
                start = chr(ord(start) + 4)
            else:
                digital_letter[str(i)] = [start, chr(ord(start) + 1), chr(ord(start) + 2)]
                start = chr(ord(start) + 3)
        print(digital_letter)

    def letterCombinations(self, digits):
        d = [" ", "*", "abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"]

        result = []

        # 递归
        # chars已组合好的字符串。  index 目前正要组合的字符的下标
        def dfs(chars, index):
            # 每次组合的结束条件
            if index == len(digits):
                result.append(chars)
                return

            cs = d[int(digits[index])]

            for c in cs:
                dfs(chars + c, index + 1)

        dfs('', 0)
        return result

    def maxProfit(self, prices):
        if len(prices) < 2:
            return 0
        # 状态机
        # profit[i][j]表示第i天,第j种状态下的利润;j=[0,1,2]分别代表:当天无任何操作,手上无股票;手上持有股票;卖出股票,手上无股票
        n = len(prices)
        profit = [[0 for i in range(3)] for i in range(n)]
        profit[0][1] = -prices[0]
        best_profit = 0
        for i in range(1, n, 1):
            profit[i][0] = max(profit[i - 1][0], profit[i - 1][2])  # 当天无操作不持股有2种情况:前一天也是无操作不持股;前一天卖股啦
            profit[i][1] = max(profit[i - 1][1],
                               profit[i - 1][0] - prices[i])  # 当天持股2种情况:前一天持股,当天无操作;当天买入了(说明昨天是没有任何操作的)
            profit[i][2] = profit[i - 1][1] + prices[i]  # 能卖出股票,说明前一天持股了
            if i == n - 1:
                best_profit = max(profit[i][0], profit[i][2])
        return best_profit


def pertition(s, l, r):
    i = l
    x = s[r]
    for j in range(l,r):
        if s[j] < x:
            s[j], s[i] = s[i], s[j]
            i += 1
    s[i], s[r] = x, s[i]
    return i

def quick_sort(s, l, r):
    if l < r:
        q = pertition(s, l, r)
        quick_sort(s, l, q-1)
        quick_sort(s, q+1, r)

class Solution:
    def maxSubArray(self, nums):
        numsum = nums[0]
        maxsum = numsum
        seq = []
        start_index = 0
        end_index = 0
        for i in range(1, len(nums)):
            if numsum < 0:
                numsum = nums[i]
                start_index = i
                end_index = i
            else:
                numsum = numsum + nums[i]
                end_index = i

            if numsum > maxsum:
                seq = nums[start_index:end_index+1]
                maxsum = numsum

        print(seq)
        print(maxsum)
        return maxsum

if __name__ == '__main__':
    # func = Solution()
    # a = func.myAtoi("   +0 123")
    # func.letterCombinations([])
    # print(func.letterCombinations('23'))
    # print(func.maxProfit([1,2,3,0,2]))

    # s = [3,5,1,0,8,9,5,4,4,8]
    # quick_sort(s, 0, len(s) - 1)
    # print(s)

    c = Solution()
    c.maxSubArray([-2,1,-3,4,-1,2,1,-5,4])

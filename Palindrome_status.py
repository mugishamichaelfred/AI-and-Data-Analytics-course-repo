def palindrome_status(strings):
    result = []   # for storing the results
    
    for string in strings:
        is_palindrome = string == string[::-1]

        # Append a tuple (string, is_palindrome) to the result list
        result.append((string, is_palindrome))
    
    return result

strings = ["racecar", "hello", "madam", "world", "level", "radar"]
print(palindrome_status(strings))

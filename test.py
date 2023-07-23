from collections import deque

# Create an empty deque
my_deque = deque(maxlen=30)
for _ in range(30):
    my_deque.append(-1)

# Add elements to the deque
my_deque.append(1)  # Add to the right end
my_deque.appendleft(2)  # Add to the left end

# Remove elements from the deque
right_element = my_deque.pop()  # Remove and return the rightmost element
left_element = my_deque.popleft()  # Remove and return the leftmost element

print(my_deque)  # Output: deque([])
print(right_element)  # Output: 1
print(left_element)  # Output: 2

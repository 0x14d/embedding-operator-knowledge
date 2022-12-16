# Explanations for File `eval_test_file_filtering`

```
A B C
	Heads: B C D F G A W X Y Z
	Tails: A B C D E F G H I J
```
H-Rank: 6<br/>
T-Rank: 3<br/>
<br/>
```
A B D
	Heads: B C D F G A W X Y Z
	Tails: A B C D E F G H I J
```
H-Rank: 6<br/>
T-Rank: 3 (C is filtered b/c of `A B C`)<br/>
<br/>
```
D E F
	Heads: D E F G H I J K L M
	Tails: F G H I J K L M N O
```
H-Rank: 1<br/>
T-Rank: 1<br/>
<br/>
```
G H I
	Heads: J K L M N O P Q R S
	Tails: J K L M N O P Q R S
```
H-Rank: not available!<br/>
T-Rank: not available!<br/>
<br/>
```
L M N
	Heads: A B P L C D E F G H
	Tails: A B O N C D E F G H
```
H-Rank: 3 (P is filtered b/c of `P M N`)<br/>
T-Rank: 3 (O is filtered b/c of `L M O`)<br/>
<br/>
```
L M O
	Heads: A B C D E F G H I J
	Tails: A B C D E F G H I J
```
H-Rank: not available!<br/>
T-Rank: not available!<br/>
<br/>
```
P M N
	Heads: A B L P E F G H I J
	Tails: A B C D E F G H I J
```
H-Rank: 3 (L is filtered b/c of `L M N`)<br/>
T-Rank: not available!<br/>
```
L M Q
	Heads: A B C D E F G H I J
	Tails: A B O N Q D E F G H
```
H-Rank: not available!<br/>
T-Rank: 3 (N and O are filtered b/c of `L M N` and `L M O`)<br/>
<br/>
```
A M N
	Heads: P L A B C D E F G H
	Tails: A B C D E F G H I J
```
H-Rank: 1 (P and L are filtered b/c of `P M N` and `L M N` )<br/>
T-Rank: not available<br/>


## Aggregate Results
```
Hits at 1: 3
Mean Rank H: (6 + 6 + 1 + 3 + 3 + 1) / 6 = 20 / 6 = 3.3333
Mean Rank T: (3 + 3 + 1 + 3 + 3) / = 13 / 5 = 2.6
Mean Rank All: (20 + 13) / (6 + 5) = 3

Total Tasks: 9 * 2 = 18
Successful Tasks H = 6
Successful Tasks T = 5
```


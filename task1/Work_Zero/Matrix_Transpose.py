if __name__ == "__mian__":
    matrix_before=[[1 for column in range(10)] for row in range(5)]
    print(f"Before transpose: {matrix_before}")
    matrix_after=[[row[column] for row in matrix_before] for column in matrix_before[0]]
    print(f"After transpose: {matrix_after}")
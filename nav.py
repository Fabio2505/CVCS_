def get_neighboring_values(grass_matrix, i, j):
    neighbors = []
    rows = len(grass_matrix)
    cols = len(grass_matrix[0])
    for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ni, nj = i + di, j + dj
        if 0 <= ni < rows and 0 <= nj < cols:
            if grass_matrix[ni][nj] is not None:
                neighbors.append(grass_matrix[ni][nj])
    return neighbors


def create_nav_grid(occlusion_matrix, grass_matrix):
    nav_matrix = []
    rows = len(occlusion_matrix)
    cols = len(occlusion_matrix[0])

    for i in range(rows):
        row = []
        for j in range(cols):
            if occlusion_matrix[i][j] == 1:
                row.append(1)  # Ostacolo
            elif grass_matrix[i][j] is None:  # cosidero il caso None solo per erba tagliata
                neighbors = get_neighboring_values(grass_matrix, i, j)
                if neighbors.count(2) > neighbors.count(0):
                    row.append(2)
                else:
                    row.append(0)
            elif grass_matrix[i][j] == 1:
                row.append(2)  # Erba già tagliata
            else:
                row.append(0)  # Erba non tagliata e libera
        nav_matrix.append(row)
    return nav_matrix

def nav_path(base_matrix, start_pos):
    """
    Naviga la matrice e restituisce il percorso principale
    e le coordinate di svuotamento finale dello stack per tornare alla posizione di partenza.
    """
    direzioni = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    percorso = []
    visitati = set()
    stack = []
    percorso_backtracking = []

    x, y = start_pos
    percorso.append((x, y))
    visitati.add((x, y))
    stack.append((x, y))

    def cella_valida(nx, ny):
        #Verifica che la cella sia dentro i limiti e non sia un ostacolo o già visitata
        return (0 <= nx < len(base_matrix) and 0 <= ny < len(base_matrix[0]) and base_matrix[nx][ny] == 0 and (nx, ny) not in visitati)

    def ci_sono_celle_da_visitare():
        #Verifica se ci sono ancora celle visitabili nella matrice
        for i in range(len(base_matrix)):
            for j in range(len(base_matrix[0])):
                if base_matrix[i][j] == 0 and (i, j) not in visitati:
                    return True
        return False

    while stack:
        trovato_prossima = False
        for dx, dy in direzioni:
            nx, ny = x + dx, y + dy
            if cella_valida(nx, ny):
                x, y = nx, ny
                percorso.append((x, y))
                visitati.add((x, y))
                stack.append((x, y))
                trovato_prossima = True
                break

        if not trovato_prossima:
            if len(stack) > 1:
                stack.pop()
                x, y = stack[-1]

                # Salva in percorso_backtracking solo se non ci sono più celle da visitare
                if not ci_sono_celle_da_visitare():
                    percorso_backtracking.append((x, y))
                else:
                    percorso.append((x, y))
            else:
                break

    return percorso, percorso_backtracking


# Eseguiamo il codice con esempio di matrice combinata, percorso e visualizzazione
occlusion_grid = [
    [0, 1, 1, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0],
    [0, 0, 0, 0, 0]
]

tag_grid = [
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
]

# Crea la matrice combinata
matrice_combinata = create_nav_grid(occlusion_grid, tag_grid)

for row in matrice_combinata:
    print(row)


# Eseguiamo il pathfinding
posizione_iniziale = (0, 0)
path, final_backtracking_path = nav_path(matrice_combinata, posizione_iniziale)
print(path)
print(final_backtracking_path)




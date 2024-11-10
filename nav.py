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
    # Definizione delle direzioni (dx, dy) per il chain code con 4 direzioni:
    # 0 = destra, 1 = giù, 2 = sinistra, 3 = su
    direzioni = [(0, 1), (1, 0), (0, -1), (-1, 0)]
    chain_code = []
    percorso = []
    visitati = set()  # todo Set per tracciare le celle già visitate oppure imposto le celle visitate a 2?
    stack = []  # Stack per il backtracking

    x, y = start_pos
    percorso.append((x, y))
    visitati.add((x, y))
    stack.append((x, y))

    def cella_valida(nx, ny):
        # Verifica che la cella sia dentro i limiti della matrice e che sia erba non tagliata (0)
        return (0 <= nx < len(base_matrix) and 0 <= ny < len(base_matrix[0]) and base_matrix[nx][ny] == 0 and (nx, ny) not in visitati)

    # Ciclo di esplorazione
    while stack:
        trovato_prossima = False
        for direzione, (dx, dy) in enumerate(direzioni):
            nx, ny = x + dx, y + dy
            if cella_valida(nx, ny):
                x, y = nx, ny
                chain_code.append(direzione)
                percorso.append((x, y))
                visitati.add((x, y))
                stack.append((x, y))
                trovato_prossima = True
                break

        if not trovato_prossima:
            # Se non ci sono celle libere adiacenti e lo stack è vuoto, termina
            if len(stack) > 1:
                stack.pop()  # Rimuovi la posizione corrente dallo stack
                x, y = stack[-1]  # Torna all'ultima posizione nello stack
                chain_code.append(-1)  # -1 come segnalazione del backtracking
                percorso.append((x, y))
            else:
                # Se lo stack ha solo un elemento (la posizione di partenza), termina
                break

    return chain_code, percorso


# Esempio di matrici
occlusion_grid = [
    [0, 1, 1],
    [0, 1, 0],
    [0, 0, 0]

]

tag_grid = [
    [0, 0, 0],
    [None, 0, 0],
    [0, 0, None]
]

matrice_combinata = create_nav_grid(occlusion_grid, tag_grid)

for row in matrice_combinata:
    print(row)

posizione_iniziale = (0, 0)

path, directions = nav_path(matrice_combinata, posizione_iniziale)

print("Percorso:", path)
print("Direzioni:", directions)






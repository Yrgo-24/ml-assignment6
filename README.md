# Inlämningsuppgift 6 - Konvolutionellt neuralt nätverk

Konvolutionellt neuralt nätverk (CNN) med utbytbara stubklasser. När stubbarna ersatts med riktiga implementationer kan nätverket tränas att känna igen och prediktera siffrorna 0 och 1 utifrån enkla tvådimensionella bilder.

## Kompilering samt körning av programmet
För att kunna kompilera koden, se till att du har `build-essential` installerat:

```bash
sudo apt -y update
sudo apt -y install build-essential
```

Tack vara den bifogade [makefilen](./makefile) kan du sedan när som helst kompilera och köra programmet via följande kommando (i denna katalog):

```bash
make
```

Du kan också enbart bygga programmet utan att köra det efteråt via följande kommando:

```bash
make build
```

Du kan köra programmet utan att kompilera innan via följande kommando:

```bash
make run
```

Du kan också ta bort kompilerade filer via följande kommando:

```bash
make clean
```

För att testa med dina implementationer av lagren i stället för stubbarna, kommentera ut flaggan `-DSTUB` under `COMPILER_FLAGS` i [makefilen](./makefile), såsom visas nedan:

```bash
COMPILER_FLAGS := -Wall -Werror -std=c++17 #-DSTUB
```
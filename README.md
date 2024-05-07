# FedFDP

Code for paper "FedFDP: Fairness-Aware Differentially Private Federated Learning"

## Dependencies

```
pip install -r requirement.txt
```

## Run

```
# generate dataset
cd ./dataset
python generate_minst.py
```

```
# start FedFDP
python mian_fdp.py -go test -dpe 1.0 -dp True -fair True -dgrdbe -True
```


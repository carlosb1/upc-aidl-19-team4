from params import BuilderTrain, MODE_DATA_SARA, SIMPLE_TRANSFORM, OPTIMIZER_ADAM, NORMAL_TRANSFORM, MODEL_SIAMESE2, MODEL_DECISION_LINEAR, MODEL_DECISION, MODEL_SIAMESE1, TRIPLET_MODEL_SIAMESE1
from train import run
import sys
PATH_DATASET = '/home/carlosb/data/cfp-dataset'


num_epochs = 100
batch_size = 12
perc_data = 1.0


BEST_MODELS = [TRIPLET_MODEL_SIAMESE1]

for model in BEST_MODELS:
    # small test for dataset sara
    print("\n")
    print("------------------TESTS for " + model + "-------------------------")

#    print(".- testing sara dataset... OPTIMIZER_SGD NORMAL_TRANSFORM v2")
#    params = BuilderTrain(PATH_DATASET).model(model).dataset(MODE_DATA_SARA, perc_data=perc_data, batch_size=batch_size).num_epochs(num_epochs).transform(NORMAL_TRANSFORM).name_run(model + "_sara_sgd_normtrans_v2").build()
#    run(params)
#    print("------")

    print(
        ".- testing sara dataset... OPTIMIZER_ADAM NORMAL_TRANSFORM LR 5e-4 v2"
    )
    params = BuilderTrain(PATH_DATASET).model(model).dataset(
        MODE_DATA_SARA, perc_data=perc_data, batch_size=batch_size).num_epochs(
            num_epochs).transform(NORMAL_TRANSFORM).name_run(
                model + "_sara_adam_normtrans_lr5e-4_v2").optimizer(
                    OPTIMIZER_ADAM, lr=5e-4, weight_decay=5e-4).build()
    run(params)
    print("------")

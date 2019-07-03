from params import BuilderTrain, MODE_DATA_SARA, SIMPLE_TRANSFORM, OPTIMIZER_ADAM, NORMAL_TRANSFORM, MODEL_SIAMESE2, MODEL_DECISION_LINEAR, MODEL_DECISION, MODEL_SIAMESE1

from train import run
import sys
if len(sys.argv) > 1 and sys.argv[1] == "aws":
    PATH_DATASET = '/root/data/cfp-dataset'
else:
    PATH_DATASET = '/home/carlosb/python-workspace/upc-aidl-19-team4/datasets/cfp-dataset'

num_epochs = 30
batch_size = 20
perc_data = 1.0

MODELS_TEST = [
    MODEL_SIAMESE1, MODEL_SIAMESE2, MODEL_DECISION_LINEAR, MODEL_DECISION
]

for model in MODELS_TEST:
    # small test for dataset sara
    print("\n")
    print("------------------TESTS for " + model + "-------------------------")
    print(".- testing sara dataset... OPTIMIZER_SGD")
    params = BuilderTrain(PATH_DATASET).model(model).dataset(
        MODE_DATA_SARA, perc_data=perc_data,
        batch_size=batch_size).num_epochs(num_epochs).transform(
            SIMPLE_TRANSFORM).name_run(model + "_sara_sgd").build()
    run(params)
    print("------")

    print(".- testing sara dataset... OPTIMIZER_ADAM")
    params = BuilderTrain(PATH_DATASET).model(model).dataset(
        MODE_DATA_SARA, perc_data=perc_data,
        batch_size=batch_size).num_epochs(num_epochs).transform(
            SIMPLE_TRANSFORM).name_run(model + "_sara_adam").optimizer(
                OPTIMIZER_ADAM, lr=1e-3, weight_decay=0.).build()
    run(params)
    print("------")

    #    print(".- testing sara dataset... OPTIMIZER_SGD NORMAL_TRANSFORM v2")
    #    params = BuilderTrain(PATH_DATASET).model(model).dataset(MODE_DATA_SARA, perc_data=perc_data, batch_size=batch_size).num_epochs(num_epochs).transform(NORMAL_TRANSFORM).name_run(model+"_sara_sgd_normtrans_v2").build()
    #    run(params)
    #    print("------")

    print(".- testing sara dataset... OPTIMIZER_ADAM NORMAL_TRANSFORM v2")

    params = BuilderTrain(PATH_DATASET).model(model).dataset(
        MODE_DATA_SARA, perc_data=perc_data,
        batch_size=batch_size).num_epochs(num_epochs).transform(
            NORMAL_TRANSFORM).name_run(model +
                                       "_sara_adam_normtrans_v2").optimizer(
                                           OPTIMIZER_ADAM,
                                           lr=1e-3,
                                           weight_decay=0.).build()
    run(params)
    print("------")
    print(
        ".- testing sara dataset... OPTIMIZER_ADAM NORMAL_TRANSFORM LR 5e-4 v2"
    )
    params = BuilderTrain(PATH_DATASET).model(model).dataset(
        MODE_DATA_SARA, perc_data=perc_data, batch_size=batch_size).num_epochs(
            num_epochs).transform(NORMAL_TRANSFORM).name_run(
                model + "_sara_adam_normtrans_lr5e-4_v2").optimizer(
                    OPTIMIZER_ADAM, lr=5e-4, weight_decay=0.).build()
    run(params)
    print("------")

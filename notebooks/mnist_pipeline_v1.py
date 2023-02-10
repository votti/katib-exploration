# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.14.0
#   kernelspec:
#     display_name: Python 3.11.0 ('katib-exp')
#     language: python
#     name: python3
# ---

# %%
import kfp
import kfp.components as components
import kfp.dsl as dsl
from kfp.components import InputPath, OutputPath, create_component_from_func

# %%
client = kfp.Client()

# %%
download_data_op = components.load_component_from_url(
    "https://raw.githubusercontent.com/kubeflow/pipelines/master/components/contrib/web/Download/component.yaml"
)

# %%
parse_mnist_op = components.load_component_from_text(
    """
name: Parse MNIST
inputs:
- {name: Images, description: gziped images in the idx format}
- {name: Labels, description: gziped labels in the idx format}
outputs:
- {name: Dataset}
metadata:
  annotations:
    author: Vito Zanotelli
    description: Based on https://github.com/kubeflow/pipelines/blob/master/components/contrib/sample/Python_script/component.yaml
implementation:
  container:
    image: tensorflow/tensorflow:2.7.1
    command:
    - sh
    - -ec
    - |
      # This is how additional packages can be installed dynamically
      python3 -m pip install pip idx2numpy
      # Run the rest of the command after installing the packages.
      "$0" "$@"
    - python3
    - -u  # Auto-flush. We want the logs to appear in the console immediately.
    - -c  # Inline scripts are easy, but have size limitaions and the error traces do not show source lines.
    - |
      import gzip
      import idx2numpy
      import sys
      from pathlib import Path
      import pickle
      import tensorflow as tf
      img_path = sys.argv[1]
      label_path = sys.argv[2]
      output_path = sys.argv[3]
      with gzip.open(img_path, 'rb') as f:
        x = idx2numpy.convert_from_string(f.read())
      with gzip.open(label_path, 'rb') as f:
        y = idx2numpy.convert_from_string(f.read())
      #one-hot encode the categories
      x_out = tf.convert_to_tensor(x)
      y_out = tf.keras.utils.to_categorical(y)
      Path(output_path).parent.mkdir(parents=True, exist_ok=True)
      with open(output_path, 'wb') as output_file:
            pickle.dump((x_out, y_out), output_file)
    - {inputPath: Images}
    - {inputPath: Labels}
    - {outputPath: Dataset}
"""
)


# %%
def process(
    data_raw_path: InputPath(str),  # type: ignore
    data_processed_path: OutputPath(str),  # type: ignore
    val_pct: float = 0.2,
    trainset_flag: bool = True,
):
    """
    Here we do all the preprocessing
    if the data path is for training data we:
    (1) Normalize the data
    (2) split the train and val data
    If it is for unseen test data, we:
    (1) Normalize the data
    This function returns in any case the processed data path
    """
    # sklearn
    from sklearn.model_selection import train_test_split
    import pickle
    import numpy as np

    def img_norm(x):
        return np.reshape(x / 255, list(x.shape) + [1])

    with open(data_raw_path, "rb") as f:
        x, y = pickle.load(f)
    if trainset_flag:

        x_ = img_norm(x)
        x_train, x_val, y_train, y_val = train_test_split(
            x_, y, test_size=val_pct, stratify=y, random_state=42
        )

        with open(data_processed_path, "wb") as output_file:
            pickle.dump((x_train, y_train, x_val, y_val), output_file)

    else:
        x_ = img_norm(x)
        with open(data_processed_path, "wb") as output_file:
            pickle.dump((x_, y), output_file)


# %%
process_op = create_component_from_func(
    func=process,
    base_image="tensorflow/tensorflow:2.7.1",  # Optional
    packages_to_install=["scikit-learn"],  # Optional
)


# %%
def train(
    data_train_path: InputPath(str),  # type: ignore
    model_out_path: OutputPath(str),  # type: ignore
    mlpipeline_metrics_path: OutputPath("Metrics"),  # type: ignore # noqa: F821
    metrics_log_path: OutputPath(str),  # type: ignore
    lr: float = 1e-4,
    optimizer: str = "Adam",
    loss: str = "categorical_crossentropy",
    epochs: int = 1,
    batch_size: int = 32,
):
    """
    This is the simulated train part of our ML pipeline where training is performed
    """

    import tensorflow as tf
    import pickle
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    import json

    with open(data_train_path, "rb") as f:
        x_train, y_train, x_val, y_val = pickle.load(f)

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Conv2D(
                64, (3, 3), activation="relu", input_shape=(28, 28, 1)
            ),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(64, (3, 3), activation="relu"),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(128, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )

    if optimizer.lower() == "sgd":
        optimizer = tf.keras.optimizers.SGD(lr)
    else:
        optimizer = tf.keras.optimizers.Adam(lr)

    model.compile(loss=loss, optimizer=optimizer, metrics=["accuracy"])

    # fit the model
    model_early_stopping_callback = tf.keras.callbacks.EarlyStopping(
        monitor="val_accuracy", patience=10, verbose=1, restore_best_weights=True
    )

    train_datagen = ImageDataGenerator(horizontal_flip=False)

    validation_datagen = ImageDataGenerator()
    history = model.fit(
        train_datagen.flow(x_train, y_train, batch_size=batch_size),
        epochs=epochs,
        validation_data=validation_datagen.flow(x_val, y_val, batch_size=batch_size),
        shuffle=False,
        callbacks=[model_early_stopping_callback],
    )

    model.save(model_out_path, save_format="tf")
    # Log accuracz
    print(history.history["accuracy"])
    print(history.history["val_accuracy"])

    metrics = {
        "metrics": [
            {
                "name": "accuracy",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": history.history["accuracy"][
                    -1
                ],  # The value of the metric. Must be a numeric value.
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
            {
                "name": "val-accuracy",  # The name of the metric. Visualized as the column name in the runs table.
                "numberValue": history.history["val_accuracy"][
                    -1
                ],  # The value of the metric. Must be a numeric value.
                "format": "PERCENTAGE",  # The optional format of the metric. Supported values are "RAW" (displayed in raw format) and "PERCENTAGE" (displayed in percentage format).
            },
        ]
    }
    with open(mlpipeline_metrics_path, "w") as f:
        json.dump(metrics, f)

    # Output metrics for Katib
    with open(metrics_log_path, "w") as f:
        f.write(f"val-accuracy={history.history['val_accuracy'][0]}\n")
        f.write(f"accuracy={history.history['accuracy'][0]}\n")
    print(metrics_log_path)


train_op = create_component_from_func(
    func=train, base_image="tensorflow/tensorflow:2.7.1", packages_to_install=["scipy"]
)


# %%
def _label_cache(step):
    """Helper to add pod cache label

    Somehow in our configuration the wrong cache label is applied :/
    """
    step.add_pod_label("pipelines.kubeflow.org/cache_enabled", "true")


# %%
@dsl.pipeline(
    name="Download MNIST dataset",
    description="A pipeline to download the MNIST dataset files",
)
def mnist_training_pipeline(
    lr: float = 1e-4,
    optimizer: str = "Adam",
    loss: str = "categorical_crossentropy",
    epochs: int = 3,
    batch_size: int = 5,
):
    TRAIN_IMG_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LAB_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

    train_imgs = download_data_op(TRAIN_IMG_URL)
    train_imgs.set_display_name("Download training images")
    _label_cache(train_imgs)

    train_y = download_data_op(TRAIN_LAB_URL)
    train_y.set_display_name("Download training labels")
    _label_cache(train_y)

    mnist_train = parse_mnist_op(train_imgs.output, train_y.output)
    mnist_train.set_display_name("Prepare train dataset")

    processed_train = process_op(mnist_train.output, val_pct=0.2, trainset_flag=True)
    processed_train.set_display_name("Preprocess images")

    training_output = (
        train_op(
            processed_train.outputs["data_processed"],
            lr=lr,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
        )
        .set_cpu_limit("1")
        .set_memory_limit("1Gi")
    )
    training_output.set_display_name("Fit the model")
    return mnist_train.output


# %%
run = client.create_run_from_pipeline_func(
    mnist_training_pipeline,
    mode=kfp.dsl.PipelineExecutionMode.V1_LEGACY,
    # You can optionally override your pipeline_root when submitting the run too:
    # pipeline_root='gs://my-pipeline-root/example-pipeline',
    arguments={},
    experiment_name="mnist",
    run_name="training_mnist_classifier_13",
    namespace="vito-zanotelli",
)


# %% [markdown]
# # Parameter tuning with Katib
#
# We now want to do parameter tuning over the pipeline with Katib.
#
# This requires:
# - adding a label to the step from which parameters should be collected
# - preventing that the step generating parameters is not skipped due to caching
#

# %%
def mnist_training_pipeline_katib(
    lr: float = 1e-4,
    optimizer: str = "Adam",
    loss: str = "categorical_crossentropy",
    epochs: int = 1,
    batch_size: int = 32,
):

    TRAIN_IMG_URL = "http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz"
    TRAIN_LAB_URL = "http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz"

    train_imgs = download_data_op(TRAIN_IMG_URL)
    train_imgs.set_display_name("Download training images")
    _label_cache(train_imgs)

    train_y = download_data_op(TRAIN_LAB_URL)
    train_y.set_display_name("Download training labels")
    _label_cache(train_y)

    mnist_train = parse_mnist_op(train_imgs.output, train_y.output)
    mnist_train.set_display_name("Prepare train dataset")

    processed_train = process_op(mnist_train.output, val_pct=0.2, trainset_flag=True)
    processed_train.set_display_name("Preprocess images")

    training_output = (
        train_op(
            processed_train.outputs["data_processed"],
            lr=lr,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            loss=loss,
        )
        .set_cpu_limit("1")
        .set_memory_limit("1Gi")
    )

    training_output.set_display_name("Fit the model")
    # This step needs to run always, as otherwise the metrics cannot be collected.
    # Other steps are cached if appropriate
    training_output.execution_options.caching_strategy.max_cache_staleness = "P0D"

    # This pod label indicates which pod Katib should collect the metric from.
    # A metrics collecting sidecar container will be added
    training_output.add_pod_label("katib.kubeflow.org/model-training", "true")


# %%
run = client.create_run_from_pipeline_func(
    mnist_training_pipeline_katib,
    mode=kfp.dsl.PipelineExecutionMode.V1_LEGACY,
    # You can optionally override your pipeline_root when submitting the run too:
    # pipeline_root='gs://my-pipeline-root/example-pipeline',
    arguments={},
    experiment_name="mnist",
    run_name="training_mnist_classifier_katib",
    namespace="vito-zanotelli",
)

# %% [markdown]
# Now setup katib

# %%
import yaml
from typing import List

import kfp.dsl as dsl
from kfp import components


from kubernetes.client.models import V1ObjectMeta
from kubeflow.katib import ApiClient
from kubeflow.katib import KatibClient
from kubeflow.katib import V1beta1Experiment
from kubeflow.katib import V1beta1ExperimentSpec
from kubeflow.katib import V1beta1AlgorithmSpec
from kubeflow.katib import V1beta1ObjectiveSpec
from kubeflow.katib import V1beta1ParameterSpec
from kubeflow.katib import V1beta1FeasibleSpace
from kubeflow.katib import V1beta1TrialTemplate
from kubeflow.katib import V1beta1TrialParameterSpec
from kubeflow.katib import V1beta1MetricsCollectorSpec
from kubeflow.katib import V1beta1CollectorSpec


# %% [markdown]
# In order to build a katib experiment, we require a trial spec.
#
# In this case the trial spec is an Argo workflow produced form the Kubeflow pipeline.
#
# The requirement to run this Argo workflow, the integration needs to be setup.
#
#
# ### Setup of Katib Argo workflow integration
# If you are running on a full Kubeflow installation *DO NOT INSTALL ARGO* as this will likely break your installation.
#
# Just run the following commands:
#
# Enable side-car injection:
#
# `kubectl patch namespace argo -p '{"metadata":{"labels":{"katib.kubeflow.org/metrics-collector-injection":"enabled"}}}'`
#
#
# Verify that the emissary executor is active (should be default in newer Kubeflow installations):
#
# ` kubectl get ConfigMap -n argo workflow-controller-configmap -o yaml | grep containerRuntimeExecutor`
#
# Patch the Katib controller:
#
# `kubectl patch ClusterRole katib-controller -n kubeflow --type=json \
#   -p='[{"op": "add", "path": "/rules/-", "value": {"apiGroups":["argoproj.io"],"resources":["workflows"],"verbs":["get", "list", "watch", "create", "delete"]}}]'
# `
#
# `kubectl patch Deployment katib-controller -n kubeflow --type=json \
#   -p='[{"op": "add", "path": "/spec/template/spec/containers/0/args/-", "value": "--trial-resources=Workflow.v1alpha1.argoproj.io"}]'`
#
# For more details and how to set this up on a partial Kubeflow installation follow:
# https://github.com/kubeflow/katib/tree/master/examples/v1beta1/argo/README.md

# %% [markdown]
# ### Helper functions to build the individual Katib Experiment Components

# %%
def create_trial_spec(pipeline, params_list: List[dsl.PipelineParam]):
    """
    Create a Katib trial specification from a KFP pipeline function

    Args:
        pipeline: a kubeflow pipeline function
        params_list (List[dsl.PipelineParam]): a list of pipeline parameters. These need
            to map the pipeline parameter to the Katib parameter.
            Eg: [dsl.PipelineParam(name='lr', value='${trialParameters.learningRate}')]

    """
    compiler = kfp.compiler.Compiler(mode=kfp.dsl.PipelineExecutionMode.V1_LEGACY)
    return compiler._create_workflow(pipeline, params_list=params_list)


# %%
def create_trial_parameters():
    """
    Defines the search space for trial parameters
    """
    # Experiment search space.
    # In this example we tune learning rate and batch size.

    parameters = [
        V1beta1ParameterSpec(
            name="learning_rate",
            parameter_type="double",
            feasible_space=V1beta1FeasibleSpace(min="0.00001", max="0.001"),
        ),
    ]
    return parameters


# %%
def create_trial_template(trial_spec):

    trial_template = V1beta1TrialTemplate(
        primary_container_name="main",  # Name of the primary container returning the metrics in the workflow
        primary_pod_labels={"katib.kubeflow.org/model-training": "true"},
        trial_parameters=[
            V1beta1TrialParameterSpec(
                name="learningRate",  # the parameter name that is replaced in your template (see Trial Specification).
                description="Learning rate for the training model",
                reference="learning_rate",  # the parameter name that experiment’s suggestion returns (parameter name in the Parameters Specification).
            )
        ],
        trial_spec=trial_spec,
        success_condition='status.[@this].#(phase=="Succeeded")#',
        failure_condition='status.[@this].#(phase=="Failed")#',
        retain=True,  # Retain completed pods - left hear for easier debugging
    )
    return trial_template


# %%
def create_metrics_collector_spec():
    """This defines the custom metrics collector"""
    return V1beta1MetricsCollectorSpec(
        source={
            "fileSystemPath": {
                "path": "/tmp/outputs/mlpipeline_metrics/data",
                "kind": "File",
            }
        },
        collector=V1beta1CollectorSpec(
            kind="Custom",
            custom_collector={
                "args": [
                    "-m",
                    "val-accuracy;accuracy",
                    "-s",
                    "katib-db-manager.kubeflow:6789",
                    "-t",
                    "$(PodName)",
                    "-path",
                    "/tmp/outputs/mlpipeline_metrics",
                ],
                "image": "votti/kfpv1-metricscollector:v0.0.10",
                "imagePullPolicy": "Always",
                "name": "custom-metrics-logger-and-collector",
                "env": [
                    {
                        "name": "PodName",
                        "valueFrom": {"fieldRef": {"fieldPath": "metadata.name"}},
                    }
                ],
            },
        ),
    )


# %%
def create_katib_experiment_spec(
    trial_spec,
    max_trial_count: int = 2,
    max_failed_trial_count: int = 1,
    parallel_trial_count: int = 2,
):
    """
    Creates the full Katib experiment

    Args:
        trial_spec: the trial specification
        max_trial_count (int): max number of trials to run, default: 2
        max_failed_trial_count (int): max number of failed trials before stopping, default: 1
        parallel_trial_count (int): max trials to run in parallel, default: 2

    Returns:
        A Katib experiment specification
    """

    # Objective specification.
    objective = V1beta1ObjectiveSpec(
        type="maximize",
        goal=0.9,
        objective_metric_name="val-accuracy",
        additional_metric_names=["accuracy"],
    )

    # Algorithm specification, see docu https://www.kubeflow.org/docs/components/katib/experiment/#search-algorithms-in-detail
    algorithm = V1beta1AlgorithmSpec(
        algorithm_name="random",
    )

    parameters = create_trial_parameters()

    # trial_spec = create_trial_spec(training_steps)

    # Configure parameters for the Trial template.
    trial_template = create_trial_template(trial_spec)

    # Metrics collector spec
    metrics_collector = create_metrics_collector_spec()

    # Create an Experiment from the above parameters.
    experiment_spec = V1beta1ExperimentSpec(
        # Experimental Budget
        max_trial_count=max_trial_count,
        max_failed_trial_count=max_failed_trial_count,
        parallel_trial_count=parallel_trial_count,
        # Optimization Objective
        objective=objective,
        # Optimization Algorithm
        algorithm=algorithm,
        # Optimization Parameters
        parameters=parameters,
        # Trial Template
        trial_template=trial_template,
        # Metrics collector
        metrics_collector_spec=metrics_collector,
    )

    return experiment_spec


# %% [markdown]
# ### Create Katib Experiment from components

# %%
trial_spec = create_trial_spec(
    mnist_training_pipeline_katib,
    params_list=[dsl.PipelineParam(name="lr", value="${trialParameters.learningRate}")],
)

# Somehow the pipeline is configured with the wrong serviceAccountName by default
trial_spec["spec"]["serviceAccountName"] = "default-editor"

# %%
katib_spec = create_katib_experiment_spec(trial_spec)

# %% [markdown]
# In order to generate a full experiment the api_version, kind and namespace need to be defined:

# %%
katib_experiment = V1beta1Experiment(
    api_version="kubeflow.org/v1beta1",
    kind="Experiment",
    metadata=V1ObjectMeta(
        name="katib-kfp-mnist-custom-32",
        namespace="vito-zanotelli",
    ),
    spec=katib_spec,
)

# %% [markdown]
# The generated yaml can written out to submit via the web ui:

# %%
with open("experiment_template_kfp_mnist_v1.yaml", "w") as f:
    yaml.dump(ApiClient().sanitize_for_serialization(katib_experiment), f)

# %% [markdown]
# Or sumitted via the KatibClient:

# %%
client = KatibClient()

# %%
client.create_experiment(katib_experiment)

# %% [markdown]
# You should now be able to observe in the Web UI how the Katib
# Experiment is running.
#
# To see how the `Argo Workflows` are started, you can also check the Kubernetes cluster:
#
# `kubectl get Workflow -n <namespace>`

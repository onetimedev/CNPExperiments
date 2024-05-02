import matplotlib.pyplot as plt
import torch
import torch.optim as optim
import collections
import matplotlib.pyplot as plt
import datetime
from DeterministicModel import DeterministicModel as Model
from GPCurvesReader import GPCurvesReader as GP
from GP import GP as GaussianProcess

def plot_functions(target_x, target_y, context_x, context_y, pred_y, var):
    """Plots the predicted mean and variance and the context points.

    Args:
      target_x: An array of shape batchsize x number_targets x 1 that contains the
          x values of the target points.
      target_y: An array of shape batchsize x number_targets x 1 that contains the
          y values of the target points.
      context_x: An array of shape batchsize x number_context x 1 that contains
          the x values of the context points.
      context_y: An array of shape batchsize x number_context x 1 that contains
          the y values of the context points.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted means of the y values at the target points in target_x.
      pred_y: An array of shape batchsize x number_targets x 1  that contains the
          predicted variance of the y values at the target points in target_x.
    """

    plot1_x = target_x[0].transpose()[0]
    plot1_y = pred_y[0].transpose()[0]

    plot2_x = plot1_x
    plot2_y = target_y[0].transpose()[0]

    plot3_x = context_x[0].transpose()[0]
    plot3_y= context_y[0].transpose()[0]

    # Plot everything
    plt.plot(plot1_x, plot1_y, "b", linewidth=2)
    plt.plot(plot2_x, plot2_y, "k:", linewidth=2)
    plt.plot(plot3_x, plot3_y, "ko", markersize=10)
    # plt.fill_between(
    #     target_x[0, :, 0],
    #     pred_y[0, :, 0] - var[0, :, 0],
    #     pred_y[0, :, 0] + var[0, :, 0],
    #     alpha=0.2,
    #     facecolor="#65c9f7",
    #     interpolate=True,
    # )

    # Make the plot pretty
    plt.yticks([-2, 0, 2], fontsize=16)
    plt.xticks([-2, 0, 2], fontsize=16)
    plt.ylim([-2, 2])
    plt.grid(False)
    # ax = plt.gca()
    # ax.set_axis_bgcolor('white')
    plt.show()


print(f"GPU Available: {torch.cuda.is_available() }")

TRAINING_ITERATIONS = int(2e5)
MAX_CONTEXt_POINTS = 10
torch.manual_seed(0)


gp_test = GaussianProcess(length_scale=0.4, amplitude=1)
tr = gp_test.build_conditional_dist()

# Train dataset
gp_train = GP(batch_size=64, max_num_context=MAX_CONTEXt_POINTS)
# Test dataset
gp_test = GP(batch_size=100, max_num_context=MAX_CONTEXt_POINTS, testing=True)


x_dimension = 1
encoder_input_dimension = 2
representation_size = 128
encoder_output_dimension = representation_size

decoder_output_dimension = 2
decoder_input_dimension = representation_size + x_dimension

hidden_layer_size = 128

# Encoder NN structure. Input Layer, Hidden Layer 1, Hidden Layer 2, Hidden Layer 3, Output Layer
encoder_sizes = [
    encoder_input_dimension,
    hidden_layer_size,
    hidden_layer_size,
    hidden_layer_size,
    encoder_output_dimension]

# Decoder NN structure. Input Layer, Hidden Layer 1, Hidden Layer 2, Output Layer
decoder_sizes = [
    decoder_input_dimension,
    hidden_layer_size,
    hidden_layer_size,
    decoder_output_dimension
]

model = Model(encoder_sizes, decoder_sizes)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

for iteration in range(TRAINING_ITERATIONS):
    data_train = gp_train.generate_curves()

    optimizer.zero_grad()
    log_prob, _, _ = model(query=data_train.query , target_y=data_train.target_y)
    loss = -log_prob.mean()
    loss.backward()
    optimizer.step()

    if iteration % 2e3 == 0:

        data_test = gp_test.generate_curves()
        test_log_prob, pred_y, var = model(data_test.query, data_test.target_y)
        test_loss = -test_log_prob.mean()

        print("{}, Iteration: {}, test loss: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), iteration, test_loss))

        (context_x, context_y), target_x = data_test.query

        plot_functions(
            target_x.detach().numpy(),
            data_test.target_y.detach().numpy(),
            context_x.detach().numpy(),
            context_y.detach().numpy(),
            pred_y.detach().numpy(),
            var.detach().numpy(),
        )


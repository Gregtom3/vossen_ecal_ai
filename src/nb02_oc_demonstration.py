import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from IPython.display import display, clear_output
import ipywidgets as widgets

def generate_plot_with_buttons(COLORS=['red'], active_buttons=[1, 2, 3, 4, 5]):
    # Global flags and data structures
    left_drawn = False   # Indicates that squares have been created
    right_drawn = False  # Indicates that latent scatter has been mapped
    minimize_loss_called = False  # Flag for bright spot effect ("Minimize coward loss")

    # Dictionaries to hold square coordinates and scatter points for each color.
    squares = {}        # Format: { color: (x, y) }
    scatter_points = {} # Format: { color: (x_array, y_array) }

    def update_plot():
        clear_output(wait=True)
        # Create figure with two subplots: left for squares, right for latent space scatter.
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))

        # Left subplot: 64x64 white background.
        white_img = np.ones((64, 64, 3))
        ax_left.set_facecolor('white')
        ax_left.imshow(white_img, extent=[0, 64, 0, 64])
        ax_left.set_xticks([])
        ax_left.set_yticks([])

        # Draw each square.
        for color in COLORS:
            if color in squares and squares[color] is not None:
                x, y = squares[color]
                rect = patches.Rectangle((x, y), 10, 10, edgecolor=color, facecolor=color, lw=2)
                ax_left.add_patch(rect)

        # Right subplot: white background with extended extents.
        ax_right.set_facecolor('white')
        ax_right.imshow(white_img, extent=[-10, 74, -10, 74])
        ax_right.set_xticks([])
        ax_right.set_yticks([])

        # Draw scatter points if latent mapping exists.
        if right_drawn and all(color in scatter_points and scatter_points[color] is not None for color in COLORS):
            for color in COLORS:
                x_vals, y_vals = scatter_points[color]
                if minimize_loss_called:
                    # First (bright) point full opacity; others lower.
                    ax_right.scatter(x_vals[0:1], y_vals[0:1], color=color, s=20, alpha=1)
                    ax_right.scatter(x_vals[1:], y_vals[1:], color=color, s=20, alpha=0.1)
                else:
                    ax_right.scatter(x_vals, y_vals, color=color, s=20)

        # Display buttons.
        buttons = [button1, button2, button3, button4, button5]
        keep_buttons = []
        for i, button in enumerate(buttons):
            if (i + 1) in active_buttons:
              keep_buttons.append(button)
        display(*keep_buttons)
        plt.show()

    def on_button1_clicked(b):
        nonlocal left_drawn, right_drawn, minimize_loss_called, squares, scatter_points
        left_drawn = True
        right_drawn = False
        minimize_loss_called = False
        # Clear any existing scatter data.
        scatter_points = {color: None for color in COLORS}
        # Generate new random positions for each square (ensuring a 10x10 square fits in 64x64).
        squares = {color: (np.random.randint(0, 64-10), np.random.randint(0, 64-10)) for color in COLORS}
        update_plot()

    def on_button2_clicked(b):
        nonlocal right_drawn, scatter_points, minimize_loss_called
        if not left_drawn or any(squares.get(color) is None for color in COLORS):
            return
        right_drawn = True
        minimize_loss_called = False
        # Generate 100 random scatter points for each color within the 64x64 region.
        scatter_points = {color: (np.random.uniform(0, 64, 100), np.random.uniform(0, 64, 100))
                          for color in COLORS}
        update_plot()

    def on_button3_clicked(b):
        nonlocal minimize_loss_called
        # "Minimize coward loss": brighten each set's first point.
        if not right_drawn or any(scatter_points.get(color) is None for color in COLORS):
            return
        minimize_loss_called = True
        update_plot()

    def on_button4_clicked(b):
        nonlocal scatter_points
        # "Minimize attractive loss": move each color's scatter points 25% closer to its own bright point.
        if not right_drawn or any(scatter_points.get(color) is None for color in COLORS):
            return
        for color in COLORS:
            x_arr, y_arr = scatter_points[color]
            bright_x, bright_y = x_arr[0], y_arr[0]
            new_x_arr = bright_x + 0.75 * (x_arr - bright_x)
            new_y_arr = bright_y + 0.75 * (y_arr - bright_y)
            scatter_points[color] = (new_x_arr, new_y_arr)
        update_plot()

    def on_button5_clicked(b):
        nonlocal scatter_points
        # "Minimize repulsive loss": for each color's scatter points, move them 5 units away from the bright spots of every other color.
        if not right_drawn or any(scatter_points.get(color) is None for color in COLORS):
            return
        for color in COLORS:
            x_arr, y_arr = scatter_points[color]
            total_disp_x = np.zeros_like(x_arr)
            total_disp_y = np.zeros_like(y_arr)
            for other in COLORS:
                if other == color:
                    continue
                other_x_arr, other_y_arr = scatter_points[other]
                # Use the bright spot (first point) of the other color as anchor.
                bright_x = other_x_arr[0]
                bright_y = other_y_arr[0]
                vec_x = x_arr - bright_x
                vec_y = y_arr - bright_y
                norms = np.sqrt(vec_x**2 + vec_y**2)
                # Avoid division by zero.
                disp_x = np.where(norms == 0, 0, 5 * vec_x / norms)
                disp_y = np.where(norms == 0, 0, 5 * vec_y / norms)
                total_disp_x += disp_x
                total_disp_y += disp_y
            new_x_arr = x_arr + total_disp_x
            new_y_arr = y_arr + total_disp_y
            scatter_points[color] = (new_x_arr, new_y_arr)
        update_plot()

    # Create buttons
    button1 = widgets.Button(description="1) Make squares", layout=widgets.Layout(width='200px'))
    button2 = widgets.Button(description="2) Map to latent space", layout=widgets.Layout(width='200px'))
    button3 = widgets.Button(description="3) Minimize coward loss", layout=widgets.Layout(width='200px'))
    button4 = widgets.Button(description="4) Minimize attractive loss", layout=widgets.Layout(width='200px'))
    button5 = widgets.Button(description="5) Minimize repulsive loss", layout=widgets.Layout(width='200px'))

    # Link buttons to their functions.
    button1.on_click(on_button1_clicked)
    button2.on_click(on_button2_clicked)
    button3.on_click(on_button3_clicked)
    button4.on_click(on_button4_clicked)
    button5.on_click(on_button5_clicked)

    update_plot()



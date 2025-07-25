# Grid World Game

A simple 2D grid world game with emoji-based visualization where you navigate an agent to reach the goal.

## 🎮 Features

- **5x5 Grid World**: Navigable 2D environment with obstacles
- **Emoji Visualization**: Clear visual representation using emojis
- **Manual Play**: Play the game yourself using WASD controls
- **Random Obstacles**: Each game has different obstacle placement

## 🎯 Game Elements

- **A** **Agent**: Your character that moves around the grid
- **G** **Goal**: The target position to reach
- **X** **Obstacles**: Barriers that block movement
- **·** **Empty Space**: Navigable areas

## 🚀 Quick Start

### Installation

1. Clone or download this repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Play the Game

Start playing immediately:

```bash
python grid_world.py
```

**Controls:**
- `W` - Move Up
- `A` - Move Left  
- `S` - Move Down
- `D` - Move Right
- `Q` - Quit

## 🎮 How to Play

1. You start at the top-left corner (🤖)
2. Navigate to the bottom-right corner (🎯)
3. Avoid obstacles (🪨) that block your path
4. Try to reach the goal in as few steps as possible!

## 🎨 Game Rules

- You can move in four directions: up, down, left, right
- You cannot move through obstacles
- You cannot move outside the grid boundaries
- The game ends when you reach the goal or hit an obstacle
- Each game has randomly placed obstacles for variety

## 🛠️ Project Structure

```
grid/
├── grid_world.py          # Main game implementation
├── requirements.txt       # Python dependencies
└── README.md             # This file
```

## 🎨 Customization

You can easily modify the game:

- **Grid Size**: Change `size` parameter in `GridWorld()`
- **Obstacles**: Modify obstacle generation in `_initialize_grid()`
- **Starting Position**: Change `agent_pos` in the constructor
- **Goal Position**: Change `goal_pos` in the constructor

## 🔬 Experiment Ideas

1. **Different Grid Sizes**: Try larger grids (10x10, 15x15)
2. **More Obstacles**: Increase the number of obstacles
3. **Different Starting Positions**: Start from different corners
4. **Time Limits**: Add a step counter or time limit
5. **Multiple Levels**: Create different obstacle patterns

## 🎯 Tips for Success

- Plan your route before moving
- Look for the shortest path to the goal
- Be careful not to get trapped by obstacles
- Each game is different due to random obstacle placement

---

**Have fun playing! 🎮** # NavBot

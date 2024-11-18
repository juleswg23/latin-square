<h2 align="center"><i>KenKen Solver</i></h2>
<h4 align="center"><i>(and generator in progress)</i></h4>


### Intro

[KenKen](https://www.kenkenpuzzle.com/) is a logic puzzle game required solvers to fill a [Latin Square](https://en.wikipedia.org/wiki/Latin_square) with values corresponding to mathematical clues in different regions. More famously, Sudoku is also a Latin Square grid.

## Table of Contents
- [Installation](#Installation)
- [Usage](#Usage)
- [Credits](#Credits)

## Installation

### Mac

To download, clone the repo onto your computer. You will need [XCode](https://apps.apple.com/us/app/xcode/id497799835?mt=12) installed and an Apple Developer account.

`git clone https://github.com/juleswg23/MegaTicTacToe`

Connect an iPhone to your mac by USB and build the project in XCode with the target being your mobile device.
Run the app on your phone (you may need to trust the developer certificate on your iPhone).

## Usage 

Both single and local multi player game modes are supported in the app.
To play locally with another player, play and pass, alternating the turns by hand. 
To play with the computer, click any square to make your move, and then press the AI move button and wait for the computer to select a move.

### Bot Model

The computer moves are backed by a monte-carlo-tree-search to converge on an optimal move.
For more details on this algorithm, check out [this version I wrote](https://github.com/juleswg23/monte-carlo-tree-search) generalized for any mutliplayer turn-based strategy game.

## Credits

This project was made possible by many sudoku [resources](https://www.sudokuwiki.org/) and KenKen solving [resouces](http://www.mlsite.net/blog/?p=95).



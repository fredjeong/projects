import pandas as pd
from collections import deque
import matplotlib.pyplot as plt
from mplfinance.original_flavor import candlestick_ohlc
import matplotlib.dates as mpl_dates
from datetime import datetime
import os
import cv2
import numpy as np

#def write_to_file(date, net_worth, filename='{}.txt'.format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
#    for i in net_worth: 
#        date += " {}".format(i)
#    #print(Date)
#    if not os.path.exists('logs'):
#        os.makedirs('logs')
#    file = open("logs/"+filename, 'a+')
#    file.write(date+"\n")
#    file.close()

class TradingGraph:
    # A crypto trading visualization using matplotlib made to render custom prices which come in following way:
    # Date, Open, High, Low, Close, Volume, net_worth, trades
    # call render every step
    def __init__(self, render_range):
        self.volume = deque(maxlen=render_range)
        self.net_worth = deque(maxlen=render_range)
        self.render_data = deque(maxlen=render_range)
        self.render_range = render_range

        # We are using the style ‘ggplot’
        plt.style.use('ggplot')
        # close all plots if there are open
        plt.close('all')
        # figsize attribute allows us to specify the width and height of a figure in unit inches
        self.fig = plt.figure(figsize=(16,8)) 

        # Create top subplot for price axis
        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
        
        # Create bottom subplot for volume which shares its x-axis
        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
        
        # Create a new axis for net worth which shares its x-axis with price
        self.ax3 = self.ax1.twinx()

        # Formatting Date
        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
        
        # Add paddings to make graph easier to view
        #plt.subplots_adjust(left=0.07, bottom=-0.1, right=0.93, top=0.97, wspace=0, hspace=0)

    # Render the environment to the screen
    def render(self, date, open, high, low, close, volume, net_worth, trades):
        # append volume and net_worth to deque list
        self.volume.append(volume)
        self.net_worth.append(net_worth)

        # before appending to deque list, need to convert Date to special format
        date = mpl_dates.date2num([pd.to_datetime(date)])[0]
        self.render_data.append([date, open, high, low, close])
        
        # Clear the frame rendered last step
        self.ax1.clear()
        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='red', colordown='blue', alpha=0.8)

        # Put all dates to one list and fill ax2 sublot with volume
        date_render_range = [i[0] for i in self.render_data]
        self.ax2.clear()
        self.ax2.fill_between(date_render_range, self.volume, 0)

        # draw our net_worth graph on ax3 (shared with ax1) subplot
        self.ax3.clear()
        self.ax3.plot(date_render_range, self.net_worth, color="blue")
        
        # beautify the x-labels (Our Date format)
        self.ax1.xaxis.set_major_formatter(self.date_format)
        self.fig.autofmt_xdate()

        # sort sell and buy orders, put arrows in appropiate order positions
        for trade in trades:
            trade_date = mpl_dates.date2num([pd.to_datetime(trade['Date'])])[0]
            if trade_date in date_render_range:
                if trade['Type'] == 'buy':
                    high_low = trade['Low']-10
                    self.ax1.scatter(trade_date, high_low, c='red', label='red', s = 120, edgecolors='none', marker="^")
                else:
                    high_low = trade['High']+10
                    self.ax1.scatter(trade_date, high_low, c='blue', label='blue', s = 120, edgecolors='none', marker="v")

        # we need to set layers every step, because we are clearing subplots every step
        self.ax2.set_xlabel('Date')
        self.ax1.set_ylabel('Price')
        self.ax3.yaxis.set_label_position('right')
        self.ax3.set_ylabel('Balance') # 여기 수정

        # I use tight_layout to replace plt.subplots_adjust
        self.fig.tight_layout()

        """Display image with matplotlib - interrupting other tasks"""
        # Show the graph without blocking the rest of the program
        #plt.show(block=False)
        # Necessary to view frames before they are unrendered
        #plt.pause(0.001)

        """Display image with OpenCV - no interruption"""
        # redraw the canvas
        self.fig.canvas.draw()
        # convert canvas to image
        img = np.fromstring(self.fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
        img  = img.reshape(self.fig.canvas.get_width_height()[::-1] + (3,))

        # img is rgb, convert to opencv's default bgr
        image = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # display image with OpenCV or any operation you like
        cv2.imshow("Bitcoin trading bot",image)

        if cv2.waitKey(25) & 0xFF == ord("q"):
            cv2.destroyAllWindows()
            return
        
def train_agent(env, visualize=False, train_episodes = 50, training_batch_size=500):
    env.create_writer() # create TensorBoard writer
    total_average = deque(maxlen=100) # save recent 100 episodes net worth
    best_average = 0 # used to track best average net worth
    for episode in range(train_episodes):
        state = env.reset(env_steps_size = training_batch_size)

        states, actions, rewards, predictions, dones, next_states = [], [], [], [], [], []
        for t in range(training_batch_size):
            env.render(visualize)
            action, prediction = env.act(state)
            next_state, reward, done = env.step(action)
            states.append(np.expand_dims(state, axis=0))
            next_states.append(np.expand_dims(next_state, axis=0))
            action_onehot = np.zeros(3)
            action_onehot[action] = 1
            actions.append(action_onehot)
            rewards.append(reward)
            dones.append(done)
            predictions.append(prediction)
            state = next_state
            
        env.replay(states, actions, rewards, predictions, dones, next_states)
        total_average.append(env.net_worth)
        average = np.average(total_average)
        
        env.writer.add_scalar('Data/average net_worth', average, episode)
        env.writer.add_scalar('Data/episode_orders', env.episode_orders, episode)
        
        print("net worth {} {:.2f} {:.2f} {}".format(episode, env.net_worth, average, env.episode_orders))
        if episode > len(total_average):
            if best_average < average:
                best_average = average
                print("Saving model")
                env.save()


def test_agent(env, visualize=True, test_episodes=10):
    env.load() # load the model
    average_net_worth = 0
    for episode in range(test_episodes):
        state = env.reset()
        while True:
            env.render(visualize)
            action, prediction = env.act(state)
            state, reward, done = env.step(action)
            if env.current_step == env.end_step:
                average_net_worth += env.net_worth
                print("net_worth:", episode, env.net_worth, env.episode_orders)
                break
            
    print("average {} episodes agent net_worth: {}".format(test_episodes, average_net_worth/test_episodes))


#import numpy as np
#import pandas as pd
#import matplotlib.pyplot as plt
#import matplotlib.dates as mpl_dates
#from collections import deque
#from mplfinance.original_flavor import candlestick_ohlc
#from datetime import datetime
#import os
#import cv2
#
#'''
#We wrote a simple render method using print statements to display the agent's net worth in our last tutorial. 
#I could have added other essential metrics, but I left them to this tutorial. 
#So, let's begin writing that logic to a new method file called utils.py to save a session's trading metrics to a file, if necessary. 
#I'll start by creating a simple function called Write_to_file(), and we'll log everything that is sent to this function to a text file:
#'''
#
#def write_to_file(date, net_worth, filename = "{}.txt".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))):
#    for i in net_worth:
#        date += " {}".format(i)
#    if not os.path.exists('logs'):
#        os.makedirs('logs')
#    file = open("logs/"+filename, 'a+')
#    file.write(date+"\n")
#    file.close
#
#class TradingGraph:
#    def __init__(self, render_range):
#        self.volume = deque(maxlen=render_range)
#        self.net_worth = deque(maxlen=render_range)
#        self.render_data = deque(maxlen=render_range)
#        self.render_range = render_range
#
#        plt.style.use('ggplot')
#        # Close all plots if there are open
#        plt.close('all')
#        # figsize attribute allows us to specify the width and height of a figure in unit inches
#        self.fig = plt.figure(figsize=(16,8))
#
#        # Create top subplot for price axis
#        self.ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
#        # Create bottom subplot for volume which shares its x-axis
#        self.ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=self.ax1)
#        # Create a new axis for net worth which shares its x-axis with price
#        self.ax3 = self.ax1.twinx()
#
#        # Formatting date
#        self.date_format = mpl_dates.DateFormatter('%d-%m-%Y')
#        
#        # Set players
#        self.ax2.set_xlabel('Date')
#        self.ax1.set_ylabel('Price')
#        self.ax3.set_ylabel('Balance')
#
#        self.fig.tight_layout()
#
#        # Show the graph with matplotlib
#        plt.show()
#    
#    def render(self, date, open, high, low, close, volume, net_worth, trades):
#        '''
#        Render the environment to the screen.
#        '''
#        self.volume.append(volume)
#        self.net_worth.append(net_worth)
#        
#        date = mpl_dates.date2num([pd.to_datetime(date)])[0]
#        self.render_data.append([date, open, high, low, close])
#
#        # Clear the frame rendered last step
#        self.ax1.clear()
#        candlestick_ohlc(self.ax1, self.render_data, width=0.8/24, colorup='red', colordown='blue', alpha=0.8)
#
#        # We need to set layers every step, because we are clearing subplots every step
#        self.ax2.set_xlabel('Date')
#        self.ax1.set_ylabel('Price')
#        self.ax3.set_ylabel('Balance')
#
#        # Show the graph without blocking the rest of the program
#        plt.show(block=False)
#        plt.pause(0.001) # Necessary to view frames before they are unrendered
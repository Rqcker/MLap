from motion_detector import df
from bokeh.plotting import figure,show,output_file
from bokeh.models import HoverTool, ColumnDataSource

# convert time to a string format
df["Start_string"]=df["Start"].dt.strftime("%Y-%m-%d %H:%M:%S")
df["End_string"]=df["End"].dt.strftime("%Y-%m-%d %H:%M:%S")

cds=ColumnDataSource(df)

p=figure(x_axis_type='datetime',height=100,width=500,sizing_mode="scale_width",title="Motion Graph")
p.yaxis.minor_tick_line_color=None
p.yaxis.ticker.desired_num_ticks=1

# the DateFrame of time values is plotted on the browser
# using Bokeh plots
hover=HoverTool(tooltips=[("Start","@Start_string"),("End","@End_string")])
p.add_tools(hover)

q=p.quad(left="Start",right="End",bottom=0,top=1,color="red",source=cds)

output_file(("graph.html"))

show(p)

import pandas
import sys
import os.path
from bokeh.plotting import figure, output_file, show, save
from bokeh.models import HoverTool
from bokeh.models import OpenURL, TapTool, CustomJS, ColumnDataSource
from bokeh.embed import autoload_static
from bokeh.layouts import layout
from bokeh.models.widgets import Select
import numpy as np

def summary_graph(report_file, plot_filename):
    output_file(plot_filename)
    df = pandas.read_csv(report_file)
    source = ColumnDataSource(data=df)
    hover = HoverTool(tooltips=[('name','@name')])
    p = figure(tools=[hover, 'tap'])
    tap = p.select(type=TapTool)
    url = "bladder_report/@name.html"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    line1 = p.circle('trap_length', 'Mean cell area', size=20, source=source)
#    line1 = p.circle(np.linspace(0,10), np.linspace(0,10))

#    show(p)
#    return

#source = ColumnDataSource(data=data)

    codex = """
            var column = cb_obj.value;
            line1.glyph.x.field = column;
            source.change.emit()
        """
    codey = """
            var column = cb_obj.value;
            line1.glyph.y.field = column;
            source.change.emit()
        """

    code_logx = """
            p.x_axis_type = cb_obj.value;
            p.change.emit()
            console.log(p);
        """

    
    code_logy = """
            p.y_axis_type = cb_obj.value;
            p.change.emit()
        """

    

    callbackx = CustomJS(args=dict(line1=line1, source=source), code=codex)
    
    callbacky = CustomJS(args=dict(line1=line1, source=source), code=codey)

    callback_logx = CustomJS(args=dict(p=p, source=source), code=code_logx)
    
    callback_logy = CustomJS(args=dict(p=p, source=source), code=code_logy)

    wx = Select(title="x-axis:", value ='total surface area', options = list(df)[1:], callback=callbackx)

    wy = Select(title="y-axis:", value ='Mean cell area', options = list(df)[1:], callback=callbacky)

#    wlx = Select(title="x-axis-type:", value ='linear', options = ['linear','log'], callback=callback_logx)

#    wly = Select(title="y-axis-type:", value ='linear', options = ['linear','log'], callback=callback_logy)

    
    save(layout([p, wx, wy]))#, wlx, wly]))
    
#summary_graph(sys.argv[1], sys.argv[2])


def summary_graph_overall(report_file, plot_filename, filenames):
    output_file(plot_filename)
    df = pandas.read_csv(report_file)
    df['url'] = [ p +'bladder_report/'+ n +'.html' for p,n in zip(filenames, df['name']) ]
    source = ColumnDataSource(data=df)
    hover = HoverTool(tooltips=[('name','@name')])
    p = figure(tools=[hover, 'tap'])
    tap = p.select(type=TapTool)
    url = "@url" #bladder_report/@name.html"
    taptool = p.select(type=TapTool)
    taptool.callback = OpenURL(url=url)
    line1 = p.circle('trap_length', 'Mean cell area', size=20, source=source)
#    line1 = p.circle(np.linspace(0,10), np.linspace(0,10))

#    show(p)
#    return

#source = ColumnDataSource(data=data)

    codex = """
            var column = cb_obj.value;
            line1.glyph.x.field = column;
            source.change.emit()
        """
    codey = """
            var column = cb_obj.value;
            line1.glyph.y.field = column;
            source.change.emit()
        """

    code_logx = """
            p.x_axis_type = cb_obj.value;
            p.change.emit()
            console.log(p);
        """

    
    code_logy = """
            p.y_axis_type = cb_obj.value;
            p.change.emit()
        """

    

    callbackx = CustomJS(args=dict(line1=line1, source=source), code=codex)
    
    callbacky = CustomJS(args=dict(line1=line1, source=source), code=codey)

    callback_logx = CustomJS(args=dict(p=p, source=source), code=code_logx)
    
    callback_logy = CustomJS(args=dict(p=p, source=source), code=code_logy)

    wx = Select(title="x-axis:", value ='total surface area', options = list(df)[1:], callback=callbackx)

    wy = Select(title="y-axis:", value ='Mean cell area', options = list(df)[1:], callback=callbacky)

#    wlx = Select(title="x-axis-type:", value ='linear', options = ['linear','log'], callback=callback_logx)

#    wly = Select(title="y-axis-type:", value ='linear', options = ['linear','log'], callback=callback_logy)

    
    save(layout([p, wx, wy]))#, wlx, wly]))
    
#summary_graph(sys.argv[1], sys.argv[2])

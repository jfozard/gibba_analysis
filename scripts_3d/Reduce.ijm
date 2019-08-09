arg = split(getArgument(),'^');
file = arg[0];
print(arg[0]);
print(arg[1]);
print(arg[2]);
print(arg[3]);
run("Bio-Formats", "open='"+file+"' autoscale color_mode=Default view=Hyperstack stack_order=XYCZT series_"+arg[2]);
Stack.getDimensions(width, height, channels, slices, frames); 
if(channels>1) {
Stack.setChannel(arg[1]);
run("Reduce Dimensionality...", "slices");
}
name=File.nameWithoutExtension;
saveAs("Tiff", arg[3]);
print("Done.");
eval("script", "System.exit(0);");
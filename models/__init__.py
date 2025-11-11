from . import interfaces

def get_model(args, img_size): 
    if args.model_name == "DAMv2":
        model = interfaces.DAMv2(img_size, args.model_size)
        return model
    elif args.model_name == "DAMv2-Metric":
        model = interfaces.DAMv2_Metric(img_size, args.model_size)
        return model
    elif args.model_name =="MonoDepth2":
        model = interfaces.MonoDepth2(img_size, args.Monodepth2_name)
        return model
    elif args.model_name =="ZoeDepth":
        model = interfaces.ZoeDepthInterface(img_size)
        return model
    else:
        raise ValueError(f"Model {args.model_name} not found") 
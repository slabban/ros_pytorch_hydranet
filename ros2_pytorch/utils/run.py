import models
import torch
import os


if __name__ == '__main__':

    isTrace = True

    hydranet =models.net(num_classes=40, num_tasks=3)

    hydranet.eval()

    if isTrace:
        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        weights_location = os.path.join(__location__,'ExpNYUD_three.ckpt')
        
        ckpt = torch.load(weights_location)
        hydranet.load_state_dict(ckpt['state_dict'])

        traced_script_module = torch.jit.trace(hydranet, torch.rand(1, 3, 480, 640))

        print(traced_script_module.code)

        traced_script_module.save("traced_nyud_hydranet.pt")

    else:

        __location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

        weights_location = os.path.join(__location__,'ExpNYUD_three.ckpt')
        
        ckpt = torch.load(weights_location)
        hydranet.load_state_dict(ckpt['state_dict'])

        script_module = torch.jit.script(hydranet)

        script_module.save("scripted_hydranet.pt")


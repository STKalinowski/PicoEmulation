import Flask

'''
Flask Application.
Load up pytorch model.
Need to setup connection and do mulltiplexing between client & server
Client -> Sends user inputs.
Server -> Sends frame image.
Continue listening till connection is closed.

Ability to reconnected, or reestablish?
I guess client is responsible for calling.
I guess there is the idea of running the model on the client,
but for now just run it on the Flask server.
'''
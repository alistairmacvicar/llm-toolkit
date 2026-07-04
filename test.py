"""
make mic
make camera
make action handler

start loop for input
  provide options for text/audio input

  if text
    type message
  if audio
    ptt to record audio
    option to re-record

  action handler needs to tell the main loop which option was chosen to select the right input
  OR???
  a main loop handler stores the last message and passes it to the llm in whatever format the input happens to be in?
  OR???
  the main loop blocks until an input choice key is pressed, while the other preview/quit/record ones live in the background?


  confirm send
  receive response
"""

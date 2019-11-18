import smtplib


fromaddr = 'nikhil.bangale@gmail.com'  
toaddrs  = 'nikhil@mapmyindia.com'  
msg = 'Spam email Test'  

username = 'nikhil.bangale@gmail.com'  
password = 'nikhil@1995'

# server = smtplib.SMTP_SSL('smtp.mapmyindia.com', 587)  
server = smtplib.SMTP_SSL('smtp.gmail.com', 465)  
server.ehlo()
server.set_debuglevel(1)
# server.starttls()
server.login(username, password)  
server.sendmail(fromaddr, toaddrs, msg)  
server.quit()


# import smtplib

# def prompt(prompt):
#     return input(prompt).strip()

# fromaddr = prompt("From: ")
# toaddrs  = prompt("To: ").split()
# print("Enter message, end with ^D (Unix) or ^Z (Windows):")

# # Add the From: and To: headers at the start!
# msg = ("From: %s\r\nTo: %s\r\n\r\n"
#        % (fromaddr, ", ".join(toaddrs)))
# while True:
#     try:
#         line = input()
#     except EOFError:
#         break
#     if not line:
#         break
#     msg = msg + line

# print("Message length is", len(msg))

# server = smtplib.SMTP('smtp.mapmyindia.com', 465)
# server.set_debuglevel(1)
# server.sendmail(fromaddr, toaddrs, msg)
# server.quit()
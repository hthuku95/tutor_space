from django.db import models
from profiles.models import UserProfile 


class OriginalPlatform(models.Model):
    platform_name = models.CharField(max_length=255)
    platform_domain_name = models.CharField(max_length=255)
    platform_home_page_url = models.URLField()
    date_created = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.platform_name


class FreelancingAccount(models.Model):
    original_platform = models.ForeignKey(OriginalPlatform, on_delete=models.CASCADE)
    user_profile = models.ForeignKey(UserProfile,on_delete=models.SET_NULL, blank=True,null=True)
    account_gmail = models.EmailField()
    recovery_gmail = models.EmailField()
    account_gmail_password = models.CharField(max_length=255)
    recovery_gmail_password = models.CharField(max_length=255)
    first_name = models.CharField(max_length=50)
    last_name = models.CharField(max_length=50)
    username = models.CharField(max_length=100)
    avatar = models.ImageField(upload_to='avatars/', blank=True, null=True)
    account_profile_url = models.URLField()
    date_created = models.DateTimeField(auto_now_add=True)
    portfolio_site_url = models.URLField()
    profile_linkedin = models.URLField(blank=True, null=True)
    profile_twitter = models.URLField(blank=True, null=True)
    security_question = models.CharField(blank=True,null=True,max_length=128)
    security_answer = models.CharField(blank=True,null=True,max_length=128)

    def __str__(self):
        return self.username
    


class File(models.Model):
    ORIGIN_CHOICES = (
        ('Assignment Submission', 'Assignment Submission'),
        ('Attachments', 'Attachments'),
        ('Revision Files', 'Revision Files'),
        ('Assignment Files', 'Assignment Files'),
    )
    file = models.FileField(upload_to='files/')
    timestamp = models.DateTimeField(auto_now_add=True)
    origin = models.CharField(max_length=30, choices=ORIGIN_CHOICES)

    def __str__(self):
        return f"File {self.id} ({self.origin})"


class Assignment(models.Model):
    ASSIGNMENT_TYPE_CHOICES = (
        ("P", "Simple Programming Tasks"),
        ("A","Academic Writing Assignments")
    )
    agent = models.ForeignKey(UserProfile, on_delete=models.CASCADE)
    original_platform = models.ForeignKey(OriginalPlatform, on_delete=models.CASCADE)
    original_account = models.ForeignKey(FreelancingAccount, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)
    subject = models.CharField(max_length=255)
    description = models.TextField()
    rates = models.DecimalField(max_digits=10, decimal_places=2)
    completion_deadline = models.DateTimeField()
    completed = models.BooleanField(default=False)
    has_revisions = models.BooleanField(default=False)
    has_deposit_been_paid = models.BooleanField(default=False)
    chat_box_url = models.CharField(max_length=255,blank=True,null=True)
    assignment_type = models.CharField(max_length=3, choices=ASSIGNMENT_TYPE_CHOICES, blank=True,null=True)

    def __str__(self):
        return self.subject

    @property
    def expected_delivery_time(self):
        time_delta = self.completion_deadline - self.timestamp
        expected_time = self.timestamp + time_delta * 0.6
        return expected_time


class AssignmentFile(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name='assignment_files')
    files = models.ManyToManyField(File)

    def __str__(self):
        return f"Assignment Files for {self.assignment.subject}"


class AssignmentSubmission(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name='submissions')
    date_completed = models.DateTimeField()
    date_to_be_delivered = models.DateTimeField()
    version = models.CharField(max_length=20)
    files = models.ManyToManyField(File)
    delivered = models.BooleanField(default=False)

    def __str__(self):
        return f"Submission {self.version} for {self.assignment.subject}"


class RevisionFile(models.Model):
    files = models.ManyToManyField(File)

    def __str__(self):
        return f"Revision File {self.id}"


class Revision(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE, related_name='revisions')
    assignment_submission = models.OneToOneField(AssignmentSubmission, on_delete=models.CASCADE)
    date_created = models.DateTimeField(auto_now_add=True)
    reason_for_submission = models.TextField()
    deadline = models.DateTimeField()
    revision_price = models.DecimalField(max_digits=10, decimal_places=2)
    has_deposit_been_paid = models.BooleanField(default=False)
    needs_payment = models.BooleanField(default=False)
    revision_files = models.ManyToManyField(RevisionFile, blank=True)

    def __str__(self):
        return f"Revision for {self.assignment_submission.assignment.subject}"

    @property
    def new_expected_time_of_delivery(self):
        time_delta = self.deadline - self.date_created
        expected_time = self.date_created + time_delta * 0.6
        return expected_time


class Chat(models.Model):
    assignment = models.ForeignKey(Assignment, on_delete=models.CASCADE)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"Chat for assignment {self.assignment.subject}"


class Message(models.Model):
    CREATOR_CHOICES = (
        (True, 'Us'),
        (False, 'Client'),
    )
    chat = models.ForeignKey(Chat, on_delete=models.CASCADE, related_name='messages')
    body = models.TextField(blank=True, null=True)
    creator = models.BooleanField(choices=CREATOR_CHOICES)
    gif_holder = models.FileField(upload_to='gifs/', blank=True, null=True)
    sticker_holder = models.FileField(upload_to='stickers/', blank=True, null=True)
    emoji_holder = models.CharField(max_length=10, blank=True, null=True)
    timestamp = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        creator = 'Us' if self.creator else 'Client'
        return f"Message by {creator} in chat {self.chat.id}"


class Attachment(models.Model):
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name='attachments')
    files = models.ManyToManyField(File)

    def __str__(self):
        return f"Attachment {self.id} for Message {self.message.id}"
    

class SearchTagPairs(models.Model):
    tag_one = models.CharField(max_length=16)
    tag_two = models.CharField(max_length=16)
    tag_three = models.CharField(max_length=16)

    def __str__(self):
        return f"{self.tag_one} {self.tag_two} {self.tag_three}"

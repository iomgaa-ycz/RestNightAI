import axios from 'axios';

async function send(url: string, data: Record<string, any>): Promise<void> {
    try {
        await axios.post(url, data);
        console.log('Instruction sent successfully');
    } catch (error) {
        console.error('Error sending instruction:', error);
    }
}

export default send;